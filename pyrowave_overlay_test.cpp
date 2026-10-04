// Copyright (c) 2026 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

// Late composition: encoding a source plus an overlay must equal encoding the
// same source with the overlay composited beforehand. Binary alpha blends
// exactly in any precision, so decoded planes must be identical; fractional
// alpha differs only by the reference's 8-bit rounding of the blend.

#include "device.hpp"
#include "context.hpp"
#include "image.hpp"

#include "pyrowave.h"
#include <stdio.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <vector>
#include <algorithm>
#include <cstdlib>

using namespace Vulkan;

#define ASSERT_THAT(x) do { \
	if (!(x)) { fprintf(stderr, "Fatal error executing %s at line %d.\n", #x, __LINE__); std::terminate(); } \
} while(false)

#define CHECKED(x) do { \
	pyrowave_result _res = x; \
	if (_res != PYROWAVE_SUCCESS) { fprintf(stderr, "Got pyrowave result %d while executing %s at line %d.\n", _res, #x, __LINE__); std::terminate(); } \
} while(false)

struct Harness
{
	Context ctx;
	Device device;
	pyrowave_device pyro = nullptr;

	Harness()
	{
		ASSERT_THAT(Context::init_loader(nullptr));
		ctx.set_num_thread_indices(1);
		ctx.set_system_handles({});
		VkApplicationInfo app_info = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
		app_info.apiVersion = VK_API_VERSION_1_3;
		app_info.pApplicationName = "pyrowave-overlay-test";
		app_info.pEngineName = "Granite";
		ctx.set_application_info(&app_info);
		ASSERT_THAT(ctx.init_instance_and_device(nullptr, 0, nullptr, 0));
		device.set_context(ctx);

		VkInstanceCreateInfo instance_create_info = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
		instance_create_info.enabledExtensionCount = device.get_device_features().num_instance_extensions;
		instance_create_info.ppEnabledExtensionNames = device.get_device_features().instance_extensions;
		instance_create_info.pApplicationInfo = &ctx.get_application_info();
		VkDeviceCreateInfo device_create_info = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
		VkDeviceQueueCreateInfo queue_info = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
		queue_info.queueFamilyIndex = device.get_queue_info().family_indices[QUEUE_INDEX_GRAPHICS];
		queue_info.queueCount = 1;
		device_create_info.pNext = ctx.get_enabled_device_features().pdf2;
		device_create_info.enabledExtensionCount = ctx.get_enabled_device_features().num_device_extensions;
		device_create_info.ppEnabledExtensionNames = ctx.get_enabled_device_features().device_extensions;
		device_create_info.queueCreateInfoCount = 1;
		device_create_info.pQueueCreateInfos = &queue_info;
		pyrowave_device_create_queue_info device_queue_info = {};
		device_queue_info.familyIndex = queue_info.queueFamilyIndex;
		device_queue_info.index = 0;
		device_queue_info.queue = device.get_queue_info().queues[QUEUE_INDEX_GRAPHICS];

		pyrowave_device_create_info info = {};
		info.GetInstanceProcAddr = vkGetInstanceProcAddr;
		info.instance = ctx.get_instance();
		info.physical_device = ctx.get_gpu();
		info.device = ctx.get_device();
		info.device_create_info = &device_create_info;
		info.instance_create_info = &instance_create_info;
		info.queue_info_count = 1;
		info.queue_info = &device_queue_info;
		info.userdata = &device;
		info.queue_lock_callback = [](void *userdata) { static_cast<Device *>(userdata)->external_queue_lock(); };
		info.queue_unlock_callback = [](void *userdata) { static_cast<Device *>(userdata)->external_queue_unlock(); };
		CHECKED(pyrowave_create_device(&info, &pyro));
	}

	~Harness()
	{
		pyrowave_device_destroy(pyro);
	}
};

struct Config
{
	bool hdr;
	bool c444;
	VkFormat overlay_format;
};

static std::vector<uint8_t> encode(Harness &h, const Config &config, int width, int height,
                                   const std::vector<uint32_t> &source, const pyrowave_overlay *overlay)
{
	pyrowave_encoder encoder = nullptr;
	pyrowave_encoder_create_info encoder_info = {};
	encoder_info.device = h.pyro;
	encoder_info.chroma = config.c444 ? PYROWAVE_CHROMA_SUBSAMPLING_444 : PYROWAVE_CHROMA_SUBSAMPLING_420;
	encoder_info.width = width;
	encoder_info.height = height;
	CHECKED(pyrowave_encoder_create(&encoder_info, &encoder));

	auto image_info = ImageCreateInfo::immutable_2d_image(width, height, VK_FORMAT_R8G8B8A8_UNORM);
	image_info.initial_layout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL;
	ImageInitialData initial = { source.data() };
	auto input = h.device.create_image(image_info, &initial);
	ASSERT_THAT(input);
	h.device.wait_idle();

	pyrowave_scaled_encode_info scaling = {};
	scaling.view.image = input->get_image();
	scaling.view.width = width;
	scaling.view.height = height;
	scaling.view.image_format = scaling.view.view_format = VK_FORMAT_R8G8B8A8_UNORM;
	scaling.view.layout = input->get_layout(VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL);
	scaling.view.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
	scaling.intermediate_plane_format = config.hdr ? VK_FORMAT_R16_UNORM : VK_FORMAT_R8_UNORM;
	scaling.input_color_space = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
	scaling.output_color_space = config.hdr ? VK_COLOR_SPACE_HDR10_ST2084_EXT : VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
	scaling.ycbcr_chroma_midpoint = config.hdr ? 512.0f / 1023.0f : 128.0f / 255.0f;
	// Dithering stays enabled: it depends only on the output position.

	pyrowave_rate_control rate_control = { size_t(width) * height * 16 + 65536 };
	if (overlay)
		CHECKED(pyrowave_encoder_encode_gpu_scaled_overlay_synchronous(encoder, nullptr, nullptr, &scaling, overlay, &rate_control));
	else
		CHECKED(pyrowave_encoder_encode_gpu_scaled_synchronous(encoder, nullptr, nullptr, &scaling, &rate_control));

	size_t num_packets = 0;
	CHECKED(pyrowave_encoder_compute_num_packets(encoder, rate_control.maximum_bitstream_size, &num_packets));
	ASSERT_THAT(num_packets == 1);
	std::vector<uint8_t> bitstream(rate_control.maximum_bitstream_size);
	pyrowave_packet packet = {};
	CHECKED(pyrowave_encoder_packetize(encoder, &packet, rate_control.maximum_bitstream_size, &num_packets,
	                                   bitstream.data(), bitstream.size()));
	bitstream.resize(packet.size);
	pyrowave_encoder_destroy(encoder);
	return bitstream;
}

// Decode to three R16 planes.
static std::vector<uint16_t> decode(Harness &h, const Config &config, int width, int height,
                                    const std::vector<uint8_t> &bitstream)
{
	pyrowave_decoder decoder = nullptr;
	pyrowave_decoder_create_info info = {};
	info.device = h.pyro;
	info.chroma = config.c444 ? PYROWAVE_CHROMA_SUBSAMPLING_444 : PYROWAVE_CHROMA_SUBSAMPLING_420;
	info.width = width;
	info.height = height;
	CHECKED(pyrowave_decoder_create(&info, &decoder));
	CHECKED(pyrowave_decoder_push_packet(decoder, bitstream.data(), bitstream.size()));
	ASSERT_THAT(pyrowave_decoder_decode_is_ready(decoder, false));

	pyrowave_gpu_buffers buffers = {};
	ImageHandle outputs[3];
	for (int c = 0; c < 3; c++)
	{
		auto output_info = ImageCreateInfo::immutable_2d_image(c && !config.c444 ? width / 2 : width,
		                                                       c && !config.c444 ? height / 2 : height,
		                                                       VK_FORMAT_R16_UNORM);
		output_info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
		output_info.initial_layout = VK_IMAGE_LAYOUT_GENERAL;
		outputs[c] = h.device.create_image(output_info);
		auto &plane = buffers.planes[c];
		plane.image = outputs[c]->get_image();
		plane.width = outputs[c]->get_width();
		plane.height = outputs[c]->get_height();
		plane.image_format = plane.view_format = VK_FORMAT_R16_UNORM;
		plane.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
		plane.layout = VK_IMAGE_LAYOUT_GENERAL;
	}
	h.device.wait_idle();
	auto cmd = h.device.request_command_buffer();
	pyrowave_device_set_command_buffer(h.pyro, cmd->get_command_buffer());
	CHECKED(pyrowave_decoder_decode_gpu_buffer(decoder, nullptr, nullptr, &buffers));
	pyrowave_device_set_command_buffer(h.pyro, VK_NULL_HANDLE);

	BufferHandle readbacks[3];
	for (int c = 0; c < 3; c++)
	{
		auto &output = *outputs[c];
		cmd->image_barrier(output, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
		                   VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
		BufferCreateInfo b = {};
		b.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		b.size = size_t(output.get_width()) * output.get_height() * sizeof(uint16_t);
		b.domain = BufferDomain::CachedHost;
		readbacks[c] = h.device.create_buffer(b);
		cmd->copy_image_to_buffer(*readbacks[c], output, 0, {}, { output.get_width(), output.get_height(), 1 },
		                          0, 0, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 });
	}
	cmd->barrier(VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
	             VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
	Fence fence;
	h.device.submit(cmd, &fence);
	fence->wait();
	std::vector<uint16_t> out;
	for (int c = 0; c < 3; c++)
	{
		auto *data = static_cast<const uint16_t *>(h.device.map_host_buffer(*readbacks[c], MEMORY_ACCESS_READ_BIT));
		out.insert(out.end(), data, data + readbacks[c]->get_create_info().size / 2);
	}
	pyrowave_decoder_destroy(decoder);
	return out;
}

static void run(Harness &h, const Config &config, bool binary_alpha)
{
	const int width = 256, height = 128;
	const int ow = 37, oh = 29, ox = -6, oy = 99; // Partially outside left/bottom edges.
	uint32_t seed = 12345;
	auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return seed >> 8; };

	std::vector<uint32_t> source(size_t(width) * height);
	for (auto &texel : source)
		texel = (rnd() & 0xffffffu) | 0xff000000u;

	// Premultiplied overlay in its own byte order.
	std::vector<uint8_t> overlay_pixels(size_t(ow) * oh * 4);
	for (int i = 0; i < ow * oh; i++)
	{
		unsigned a = binary_alpha ? ((rnd() & 1) ? 255 : 0) : (rnd() & 0xff);
		uint8_t *t = &overlay_pixels[size_t(i) * 4];
		for (int c = 0; c < 3; c++)
			t[c] = uint8_t(rnd() % (a + 1));
		t[3] = uint8_t(a);
	}
	const bool bgra = config.overlay_format == VK_FORMAT_B8G8R8A8_UNORM;

	auto composited = source;
	for (int y = 0; y < oh; y++)
	{
		for (int x = 0; x < ow; x++)
		{
			int px = ox + x, py = oy + y;
			if (px < 0 || py < 0 || px >= width || py >= height)
				continue;
			const uint8_t *o = &overlay_pixels[(size_t(y) * ow + x) * 4];
			uint8_t rgba_o[4] = { bgra ? o[2] : o[0], o[1], bgra ? o[0] : o[2], o[3] };
			auto &d = composited[size_t(py) * width + px];
			uint32_t out = 0xff000000u;
			for (int c = 0; c < 3; c++)
			{
				float dst = float((d >> (8 * c)) & 0xff) / 255.0f;
				float v = rgba_o[c] / 255.0f + dst * (1.0f - rgba_o[3] / 255.0f);
				out |= uint32_t(std::lround(std::min(1.0f, std::max(0.0f, v)) * 255.0f)) << (8 * c);
			}
			d = out;
		}
	}

	pyrowave_overlay overlay = {};
	overlay.pixels = overlay_pixels.data();
	overlay.width = ow;
	overlay.height = oh;
	overlay.stride = ow * 4;
	overlay.format = config.overlay_format;
	overlay.x = ox;
	overlay.y = oy;
	overlay.generation = 1;

	// The overlay is a specialization of the scaler; the driver may make
	// different relaxed-precision choices in that variant. The reference
	// therefore runs the same variant with an inert (off-input) overlay, so
	// the comparison isolates the blend itself.
	pyrowave_overlay inert = overlay;
	inert.x = width + 1;
	auto late = encode(h, config, width, height, source, &overlay);
	auto early = encode(h, config, width, height, composited, &inert);
	auto plain = encode(h, config, width, height, source, nullptr);
	auto plain_variant = encode(h, config, width, height, source, &inert);
	ASSERT_THAT(late != plain);
	// Payload placement uses atomics, so bitstream layout differs between
	// encodes of identical input; decoded samples are deterministic.
	auto plain_decoded = decode(h, config, width, height, plain);
	ASSERT_THAT(plain_decoded == decode(h, config, width, height, encode(h, config, width, height, source, nullptr)));
	// Bound the variant's own precision effect against the plain shader.
	{
		auto variant = decode(h, config, width, height, plain_variant);
		double se = 0.0;
		unsigned worst = 0;
		for (size_t i = 0; i < variant.size(); i++)
		{
			int d = int(variant[i]) - int(plain_decoded[i]);
			se += double(d) * d;
			worst = std::max(worst, unsigned(std::abs(d)));
		}
		double psnr = se == 0.0 ? 999.0 : 10.0 * std::log10(65535.0 * 65535.0 * variant.size() / se);
		printf("%s %s overlay variant vs plain shader: max %u/65535, PSNR %.1f dB\n", config.hdr ? "PQ2020/R16" : "SDR709/R8",
		       config.c444 ? "444" : "420", worst, psnr);
		ASSERT_THAT(psnr > 60.0);
	}

	const char *name = config.hdr ? "PQ2020/R16" : "SDR709/R8";
	auto a = decode(h, config, width, height, late);
	auto b = decode(h, config, width, height, early);
	if (binary_alpha)
	{
		bool identical = a == b;
		if (!identical)
		{
			size_t diffs = 0, worst_at = 0;
			unsigned worst = 0;
			for (size_t i = 0; i < a.size(); i++)
				if (a[i] != b[i])
				{
					diffs++;
					unsigned d = unsigned(std::abs(int(a[i]) - int(b[i])));
					if (d > worst) { worst = d; worst_at = i; }
				}
			printf("  %zu/%zu samples differ, worst %u at plane offset %zu (y=%zu x=%zu)\n", diffs, a.size(), worst,
			       worst_at, worst_at / width, worst_at % width);
		}
		printf("%s %s %s binary alpha: decoded planes %s\n", name, config.c444 ? "444" : "420",
		       bgra ? "BGRA" : "RGBA", identical ? "identical" : "DIFFER");
		fflush(stdout);
		ASSERT_THAT(identical);
	}
	else
	{
		unsigned worst = 0;
		for (size_t i = 0; i < a.size(); i++)
			worst = std::max(worst, unsigned(std::abs(int(a[i]) - int(b[i]))));
		printf("%s %s %s fractional alpha: max decoded difference %u/65535\n", name, config.c444 ? "444" : "420",
		       bgra ? "BGRA" : "RGBA", worst);
		// The reference rounds each blended texel to 8 bits before encoding;
		// allow that rounding (half a code, propagated through the codec).
		ASSERT_THAT(worst <= 3u * 257u);
	}
}

int main()
{
	Harness h;
	for (bool hdr : { false, true })
		for (bool c444 : { false, true })
			for (VkFormat format : { VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_B8G8R8A8_UNORM })
				for (bool binary : { true, false })
					run(h, { hdr, c444, format }, binary);

	// Unsupported requests fail before submitting anything.
	pyrowave_encoder encoder = nullptr;
	pyrowave_encoder_create_info encoder_info = {};
	encoder_info.device = h.pyro;
	encoder_info.chroma = PYROWAVE_CHROMA_SUBSAMPLING_444;
	encoder_info.width = 128;
	encoder_info.height = 64;
	CHECKED(pyrowave_encoder_create(&encoder_info, &encoder));
	uint32_t texel = 0xffffffffu;
	pyrowave_overlay overlay = { &texel, 1, 1, 4, VK_FORMAT_R8G8B8A8_UNORM, 0, 0, 1 };
	pyrowave_scaled_encode_info scaling = {};
	scaling.view.width = 256; // Scaled input cannot be composited 1:1.
	scaling.view.height = 128;
	scaling.view.view_format = VK_FORMAT_R8G8B8A8_UNORM;
	scaling.intermediate_plane_format = VK_FORMAT_R8_UNORM;
	pyrowave_rate_control rate_control = { 1 << 20 };
	ASSERT_THAT(pyrowave_encoder_encode_gpu_scaled_overlay_synchronous(encoder, nullptr, nullptr, &scaling, &overlay,
	                                                                  &rate_control) == PYROWAVE_ERROR_INVALID_ARGUMENT);
	overlay.format = VK_FORMAT_R16G16B16A16_SFLOAT;
	scaling.view.width = 128;
	scaling.view.height = 64;
	ASSERT_THAT(pyrowave_encoder_encode_gpu_scaled_overlay_synchronous(encoder, nullptr, nullptr, &scaling, &overlay,
	                                                                  &rate_control) == PYROWAVE_ERROR_INVALID_ARGUMENT);
	pyrowave_encoder_destroy(encoder);
	printf("overlay tests passed\n");
}
