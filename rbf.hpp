#include <vector>

using namespace std;


// structure for neurons of secret layer
typedef struct {
    vector<float> weights;
    float sigma;
    float output;
} secret_neuron_t;


// strukture for neurons of output layer
typedef struct {
    vector<float> weights;
    float output;
} output_neuron_t;


// structure 
typedef struct {
    vector<float> coordinates;
    unsigned cls;
    unsigned nearest_neuron;
} input_data;


// create and init secret layer of q neurons
vector<secret_neuron_t> init_secret_layer(vector<input_data> in, unsigned q, bool rnd);


// loads training data to a vector
vector<input_data> get_data(const char *filename, unsigned dim, unsigned cls);


// calculate neareast neuron for each vector of data
bool calculate_nearest_neurons(vector<input_data> *data, vector<secret_neuron_t> secret_layer);


// calculate centers of neurons
void calculate_center(vector<secret_neuron_t> *secret_layer, vector<input_data> data);


// calculate sigma for each neuron of secret layer
void calculate_sigma(vector<secret_neuron_t> *secret_layer, vector<input_data> data);


// create and initialize output layer of cls neurons with q neurons in previous layer
vector<output_neuron_t> init_output_layer(unsigned q, unsigned cls, bool rnd);


// count number of correctly classified data vectors at the moment
unsigned count_correct(vector<output_neuron_t> output_layer, vector<secret_neuron_t> secret_layer, vector<input_data> data);


// simulate training of neurons
void train(vector<output_neuron_t> output_layer, vector<secret_neuron_t> secret_layer, vector<input_data> data, float mi);
