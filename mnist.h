#pragma once
#include <torch/torch.h>
#include <torch/csrc/jit/serialization/import.h>
#include <opencv2/opencv.hpp>


#include <string>
#include <vector>


void train_mnist();
void test_model();
void test_jitmodel();

std::vector<std::tuple<torch::Tensor, torch::Tensor>> ReadRsv(const std::string path);

class MnistDataset : public  torch::data::Dataset<MnistDataset>
{
private:
	std::vector<std::tuple<torch::Tensor, torch::Tensor>> rsv_;
public:
	MnistDataset(std::string& file_names_rsv)
	{
		rsv_ = ReadRsv(file_names_rsv);
	};
	torch::data::Example<> get(size_t index) override
	{
		torch::Tensor line = std::get<0>(rsv_[index]);
		torch::Tensor label = std::get<1>(rsv_[index]);

		return { line,label };
	};
	torch::optional<size_t> size() const override
	{
		return rsv_.size();
	};
};



struct Mnistmodel : torch::nn::Module {
	Mnistmodel()
		:conv1(torch::nn::Conv2dOptions(1, 10, 5)),
		conv2(torch::nn::Conv2dOptions(10, 20, 5)),
		dropout1(0.5),
		fc1(320, 50),
		fc2(50, 10)
	{
		register_module("conv1", conv1);
		register_module("conv2", conv2);
		register_module("dropout1", dropout1);
		//register_module("dropout2", dropout2);
		register_module("fc1", fc1);

		register_module("fc2", fc2);
	}
	torch::Tensor forward(torch::Tensor x) {
		x = conv1(x);
		x = torch::max_pool2d(x, { 2,2 });
		x = torch::relu(x);
		x = conv2(x);
		x = dropout1(x);
		x = torch::max_pool2d(x, { 2,2 });
		x = torch::relu(x);
		x = x.view({ -1,320 });
		x = fc1(x);
		x = torch::relu(x);

		x = torch::dropout(x, 0.5, is_training());
		x = fc2(x);
		return torch::log_softmax(x, /*dim=*/1);
	}
	torch::nn::Conv2d conv1;
	torch::nn::Conv2d conv2;
	torch::nn::Dropout2d dropout1;
	torch::nn::Linear fc1;
	torch::nn::Linear fc2;
};



struct Options {
	size_t epochs;
	size_t batch_size;//批大小
	size_t test_batch_size;
	float lr;
	float momentum;
	uint64_t seed;
	size_t log_interval;
	bool save_model;
	bool no_cuda;
	std::string train_rsv;
	std::string test_rsv;
};


template <typename DataLoader>
void train(int32_t epoch, const Options& options, std::shared_ptr<Mnistmodel> model, torch::Device device, DataLoader& data_loader, torch::optim::Optimizer& optimizer, size_t dataset_size)
{
	model->train();
	size_t batch_idx = 0;
	for (auto& batch : data_loader) {
		auto data = batch.data.to(device);
		auto targets = batch.target.to(device);
		targets = targets.reshape(options.batch_size);
		optimizer.zero_grad();
		auto output = model->forward(data);
		auto loss = torch::nll_loss(output, targets);
		loss.backward();
		optimizer.step();
		batch_idx++;
		if ((batch_idx * batch.data.size(0)) % options.log_interval == 0)
		{
			std::printf(
				"\r训练批次: %ld [%5ld/%5ld] Loss: %.6f",
				//"训练批次: %ld [%5ld/%5ld] Loss: %.6f\r\n",
				epoch,
				batch_idx * batch.data.size(0),
				dataset_size,
				loss.item<float>());
		}

	}
}

template <typename DataLoader>
void test(const Options& options, std::shared_ptr<Mnistmodel> model, torch::Device device, DataLoader& data_loader, size_t dataset_size)
{
	model->eval();
	float test_loss = 0;
	int64_t correct = 0;
	for (const auto& batch : data_loader) {
		auto data = batch.data.to(device);
		auto targets = batch.target.to(device);
		targets = targets.reshape(options.test_batch_size);
		auto output = model->forward(data);
		test_loss += torch::nll_loss(output, targets,/*weight=*/{}, torch::Reduction::Sum).item<float>();
		auto pred = output.argmax(1);
		correct += pred.eq(targets).sum().item<int64_t>();
	}
	test_loss /= dataset_size;
	std::printf(
		"\n测试: 平均 loss: %.6f | 置信率: %.6f\n",
		//"测试:平均 loss: %.6f | 置信率: %.6f;\r\n",
		test_loss,
		static_cast<float>(correct) / dataset_size);
}

template <typename DataLoader>
void testmnistjit(int64_t test_batch_size, std::shared_ptr<torch::jit::script::Module> model, torch::Device device, DataLoader& data_loader, size_t dataset_size)
{
	model->eval();
	float test_loss = 0;
	int64_t correct = 0;
	for (const auto& batch : data_loader) {
		auto data = batch.data.to(device);
		auto targets = batch.target.to(device);
		targets = targets.reshape(test_batch_size);
		std::vector<torch::jit::IValue> jitdata;
		jitdata.push_back(data);
		auto output = model->forward(jitdata).toTensor();
		//std::cout << output << std::endl;
		test_loss += torch::nll_loss(output, targets,/*weight=*/{}, torch::Reduction::Sum).item<float>();
		auto pred = output.argmax(1);
		correct += pred.eq(targets).sum().item<int64_t>();
	}

	test_loss /= dataset_size;
	std::printf(
		"\n测试: 平均 loss: %.6f | 置信率: %.6f\n",
		test_loss,
		static_cast<float>(correct) / dataset_size);
}