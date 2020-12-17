#include <torch/torch.h>
#include "mnist.h"
int main()
{	
	/*std::string path = "F:/deeplearning/models/train.rsv";
	std::ifstream fs(path, std::ios::binary);
	int cout;
	int length;
	fs.read(reinterpret_cast<char*>(&cout), sizeof(int));
	std::cout << cout << std::endl;
	std::vector<byte> datac;
	std::vector<byte> datad;
	datac.resize(784);
	datad.resize(1);
	for (int i = 0; i < cout; i++) {
		fs.read(reinterpret_cast<char*>(&length), sizeof(int));	
		fs.read(reinterpret_cast<char*>(datac.data()), 784);
		fs.read(reinterpret_cast<char*>(datad.data()), 1);
	}
	std::cout << "length:" << length << std::endl;*/
	//auto t = torch::tensor({ {1,2,3,0,0,0,0},{3,4,5,6,0,0,0} });
	//auto aa = torch::zeros({ 2 }, torch::kInt64);

	//aa.index_put_({ 0 }, torch::count_nonzero(t[0]));
	//aa.index_put_({ 1 }, torch::count_nonzero(t[1]));
	////aa[0]= torch::count_nonzero(t[0]) ;
	////aa[1] = torch::count_nonzero(t[1]);
	//auto b = torch::masked_select(t, t > 0);
	//std::cout << aa << std::endl;
	//std::cout << b<<std::endl;
	//train_mnist();
	//test_model();
	
	//test_jitmodel();
}
