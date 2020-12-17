#include "mnist.h"

std::vector<std::tuple<torch::Tensor, torch::Tensor>> ReadRsv(const std::string path) {
	int cout;
	int length;
	std::ifstream fs(path, std::ios::binary);
	fs.read(reinterpret_cast<char*>(&cout), sizeof(int));
	std::vector<std::tuple<torch::Tensor, torch::Tensor>> rsv;
	torch::Tensor line;
	torch::Tensor label;
	std::vector<uchar> datac;
	std::vector<uchar> datad;
	datac.resize(784);
	datad.resize(1);
	for (int i = 0; i < cout; i++) {
		fs.read(reinterpret_cast<char*>(&length), sizeof(int));
		fs.read(reinterpret_cast<char*>(datac.data()), 784);
		fs.read(reinterpret_cast<char*>(datad.data()), 1);
		line = torch::from_blob(datac.data(), { 784 }, torch::kByte);
		line = line.reshape({ 1,28,28 }).toType(torch::kFloat);
		label = torch::from_blob(datad.data(), { 1 }, torch::kByte);
		label = label.reshape({ 1 }).toType(torch::kLong);
		rsv.push_back(std::make_tuple(line, label));
	}
	return rsv;
}

void train_mnist() {
	torch::manual_seed(1);
	Options options;
	options.epochs = 10;
	options.batch_size = 1000;//批大小
	options.test_batch_size = 1000;
	options.lr = 0.01;
	options.momentum = 0.5;
	options.seed = 1;
	options.log_interval = 1000;
	options.save_model = false;
	options.no_cuda = true;
	options.train_rsv = "F:/deeplearning/models/train.rsv";
	options.test_rsv = "F:/deeplearning/models/test.rsv";
	auto device_type = torch::kCUDA;
	auto device = torch::Device(device_type);
	auto net = std::make_shared<Mnistmodel>();
	net->to(device);
	auto train_dataset = MnistDataset(options.train_rsv).map(torch::data::transforms::Stack<>());

	const auto dataset_size = train_dataset.size().value();
	auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(train_dataset), options.batch_size);
	auto test_dataset = MnistDataset(options.test_rsv).map(torch::data::transforms::Stack<>());

	const size_t test_dataset_size = test_dataset.size().value();
	auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(test_dataset), options.test_batch_size);
	//torch::optim::SGD optimizer(net->parameters(), torch::optim::SGDOptions(options.lr).momentum(0.5));
	torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(options.lr).betas({ 0.9,0.999 }));

	std::cout << "开始训练" << std::endl;
	auto t1 = std::chrono::steady_clock::now();
	for (size_t epoch = 1; epoch <= options.epochs; ++epoch) {
		train(epoch, options, net, device, *train_loader, optimizer, dataset_size);
		test(options, net, device, *test_loader, test_dataset_size);
	}
	std::string savepath = "F:/deeplearning/models/mnist.pt";
	torch::save(net, savepath);
	auto t2 = std::chrono::steady_clock::now();
	std::cout << "耗时(秒):" << std::chrono::duration<double, std::milli>(t2 - t1).count() / 1000 << std::endl;

}

void test_model() {
	torch::manual_seed(1);
	Options options;
	options.epochs = 1;
	options.batch_size = 1;//批大小
	options.test_batch_size = 1;
	options.lr = 0.01;
	options.momentum = 0.5;
	options.seed = 1;
	options.log_interval = 500;
	options.save_model = false;
	options.no_cuda = true;

	options.test_rsv = "../models/test.rsv";
	auto device_type = torch::kCPU;
	auto device = torch::Device(device_type);
	std::string savepath = "../models/mnist.pt";
	auto net = std::make_shared<Mnistmodel>();
	torch::load(net, savepath);
	net->to(device);
	auto test_dataset = MnistDataset(options.test_rsv).map(torch::data::transforms::Stack<>());
	const size_t test_dataset_size = test_dataset.size().value();
	auto test_loader = torch::data::make_data_loader(std::move(test_dataset), options.test_batch_size);
	auto t1 = std::chrono::steady_clock::now();
	for (size_t epoch = 1; epoch <= 1; ++epoch) {
		test(options, net, device, *test_loader, test_dataset_size);
	}
	auto t2 = std::chrono::steady_clock::now();
	std::cout << "耗时(秒):" << std::chrono::duration<double, std::milli>(t2 - t1).count() / 1000 << std::endl;
}

void test_jitmodel() {

	torch::manual_seed(1);
	Options options;
	options.epochs = 1;
	options.batch_size = 500;//批大小
	options.test_batch_size = 1000;
	options.lr = 0.01;
	options.momentum = 0.5;
	options.seed = 1;
	options.log_interval = 500;
	options.save_model = false;
	options.no_cuda = false;
	options.train_rsv = "../models/train.rsv";
	options.test_rsv = "../models/test.rsv";
	auto device_type = torch::kCPU;
	auto device = torch::Device(device_type);
	std::string savepath = "../models/jit_mnist1.pth";
	std::cout << "开始加载" << std::endl;
	torch::jit::Module module = torch::jit::load(savepath);
	std::cout << "加载完毕" << std::endl;
	auto net = std::make_shared<torch::jit::Module>(module);
	net->to(device);
	auto test_dataset = MnistDataset(options.test_rsv).map(torch::data::transforms::Stack<>());
	//.map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
	const size_t test_dataset_size = test_dataset.size().value();
	auto test_loader = torch::data::make_data_loader(std::move(test_dataset), options.test_batch_size);
	auto t1 = std::chrono::steady_clock::now();
	for (size_t epoch = 1; epoch <= 1; ++epoch) {
		testmnistjit(options.test_batch_size, net, device, *test_loader, test_dataset_size);
	}
	auto t2 = std::chrono::steady_clock::now();
	std::cout << "耗时(秒):" << std::chrono::duration<double, std::milli>(t2 - t1).count() / 1000 << std::endl;
}