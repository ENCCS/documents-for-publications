// Iterate over all available devices
for (const auto &device : sycl::device::get_devices())
{
    // Print the device name
    std::cout << "Creating a queue on " << device.get_info<sycl::info::device::name>() << "\n";

    // Create an in-order queue for the current device
    sycl::queue q(device, {sycl::property::queue::in_order()});

    // Now we can submit tasks to q!
}
