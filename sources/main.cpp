/*
	*This is main file which starts and 'teech' a neuronet and show you logs
*/

#include "trainingSet.h"
#include "neuron.h"
#include "net.h"


void showVectorVals(std::string label, std::vector<double> &v) {
	std::cout << label << " ";
	for (unsigned i = 0; i < v.size(); ++i) {
		std::cout << v[i] << " ";
	}
}




int main(void) {
	TrainingSet trainingData("data.txt");
	std::vector<unsigned> topology;
	trainingData.getTopology(topology);
	Net net(topology);
	std::vector<double> inputVals, targetVals, resultVals;
	int trainingPass = 0;


	while (!trainingData.isEOF()) {
		++trainingPass;
		std::cout << std::endl << "Pass: " << trainingPass << std::endl;

		if (trainingData.getNextInputs(inputVals) != topology[0]) break;

		showVectorVals("Input:", inputVals);
		std::cout << std::endl;
		net.feedForward(inputVals);

		trainingData.getTargetOutputs(targetVals);
		assert(targetVals.size() == topology.back());

		net.getResults(resultVals);
		showVectorVals("Outputs:", resultVals);
		showVectorVals("=>", targetVals);
		std::cout << std::endl;

		net.backProp(targetVals);

		std::cout << "Net average error: " << net.getRecentAverageError() << std::endl;
	}
	std::cout << std::endl << "Done" << std::endl;

	return 0;
}
