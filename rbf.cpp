#include <iostream>
#include <iomanip>
#include <string>
#include <string.h>
#include <vector>
#include <getopt.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <ctime>
#include <limits>
#include <cmath>
#include "rbf.hpp"

using namespace std;


// create and init secret layer of q neurons
vector<secret_neuron_t> init_secret_layer(vector<input_data> in, unsigned q, bool rnd)
{
    vector<secret_neuron_t> secret_layer;
    vector<int> v;

    // random initialization of weights
    if (rnd) {
        for (unsigned i = 0; i < q; i++) {
            int x = rand() % in.size();
            while (find(v.begin(), v.end(), x) != v.end())
                x = rand() % in.size();

            v.push_back(x);
        }
    // initialization of weight from first data
    } else {
        for (unsigned i = 0; i < q; i++) {
            v.push_back(i);
        }
    }

    for (int i : v) {
        secret_neuron_t n;
        n.weights = in[i].coordinates;
        secret_layer.push_back(n);
    }

    return secret_layer;
}


// loads training data to a vector
vector<input_data> get_data(const char *filename, unsigned dim, unsigned cls)
{
    ifstream instrm(filename);
    vector<input_data> data;
    float coordinate;
    
    for (string line; getline(instrm, line); ) {
        istringstream in(line);
        input_data d;

        for (unsigned i = 0; i < dim; i++) {
            in >> coordinate;
            d.coordinates.push_back(coordinate);
        }
        in >> d.cls;
        d.cls = d.cls > cls ? cls : d.cls;

        data.push_back(d);
    }

    return data;
}


// calculate neareast neuron for each vector of data
bool calculate_nearest_neurons(vector<input_data> *data, vector<secret_neuron_t> secret_layer)
{
    bool change = false;
    unsigned nearest = 0;

    for (unsigned i = 0; i < data->size(); i++) {
        for (unsigned k = 0; k < (*data)[i].coordinates.size(); k++)
            cout << fixed << setw(6) << setprecision(2) << (*data)[i].coordinates[k] << " ";
        cout << "  | ";
        
        float min = numeric_limits<float>::infinity();
        for (unsigned j = 0; j < secret_layer.size(); j++) {
            float sum = 0;
            for (unsigned k = 0; k < (*data)[i].coordinates.size(); k++) {
                float tmp = (*data)[i].coordinates[k] - secret_layer[j].weights[k];
                sum += tmp  * tmp;
            }
            sum = sqrtf(sum);

            if (sum < min) {
                min = sum;
                nearest = j;
            }
            cout << fixed << setw(6) << setprecision(2) << sum << " ";
        }
        
        change = change || (*data)[i].nearest_neuron != nearest;
        (*data)[i].nearest_neuron = nearest;
        cout << "  |   " << min << " (" << (*data)[i].nearest_neuron << ")" << endl;
    }

    return change;
}


// calculate centers of neurons
void calculate_center(vector<secret_neuron_t> *secret_layer, vector<input_data> data)
{
    for (unsigned i = 0; i < secret_layer->size(); i++) {
        int n = 0;
        vector<float> weights;
        for (unsigned j = 0; j < (*secret_layer)[i].weights.size(); j++) {
            weights.push_back(0);
        }

        for (unsigned j = 0; j < data.size(); j++) {
            if (data[j].nearest_neuron == i) {
                n++;
                for (unsigned k = 0; k < data[j].coordinates.size(); k++) {
                    weights[k] += data[j].coordinates[k];
                }
            }
        }
        transform(weights.begin(), weights.end(), weights.begin(), [n](float &c){ return c/n; });
        if ((*secret_layer)[i].weights != weights) {
            cout << "neuron " << i << ":";
            for (unsigned j = 0; j < (*secret_layer)[i].weights.size(); j++) {
                cout << " " << fixed << setw(6) << setprecision(2) << (*secret_layer)[i].weights[j];
            }
            cout << " ->";
            for (unsigned j = 0; j < (*secret_layer)[i].weights.size(); j++) {
                cout << " " << fixed << setw(6) << setprecision(2) << weights[j];
            }
            cout << endl;
            (*secret_layer)[i].weights = weights;
        }
    }
}


// calculate sigma for each neuron of secret layer
void calculate_sigma(vector<secret_neuron_t> *secret_layer, vector<input_data> data)
{
    // sigma based on distance of vectors of data in cluster
    for (unsigned i = 0; i < secret_layer->size(); i++) {
        unsigned n = 0;
        float dist = 0;
        for (input_data d: data) {
            if (d.nearest_neuron == i) {
                n++;
                for (unsigned k = 0; k < d.coordinates.size(); k++) {
                    float tmp = d.coordinates[k] - (*secret_layer)[i].weights[k];
                    dist += tmp * tmp;
                }
            }
        }
        (*secret_layer)[i].sigma = n > 1 ? sqrtf(dist / (n - 1)) : 0.f;
        cout << "neuron " << i << " - sigma: " << (*secret_layer)[i].sigma << endl;
    }
}


// create and initialize output layer of cls neurons with q neurons in previous layer
vector<output_neuron_t> init_output_layer(unsigned q, unsigned cls, bool rnd)
{
    vector<output_neuron_t> output_layer;
    for (unsigned i = 0; i < cls; i++) {
        output_neuron_t n;
        for (unsigned j = 0; j < q; j++) {
            // random initialization with <-1;1>
            if (rnd) {
                n.weights.push_back(1.f - 2.f * rand() / numeric_limits<int>::max());
            // zero initialization
            } else {
                n.weights.push_back(0.f);
            }
        }

        output_layer.push_back(n);
    }

    return output_layer;
}


// count number of correctly classified data vectors at the moment
unsigned count_correct(vector<output_neuron_t> output_layer, vector<secret_neuron_t> secret_layer, vector<input_data> data)
{
    unsigned answer = 0;
    unsigned correct = 0;
    for (input_data d : data) {
        // outputs of secret layer for data d
        for (unsigned i = 0; i < secret_layer.size(); i++) {
            secret_layer[i].output = 0.f;
            for (unsigned j = 0; j < secret_layer[i].weights.size(); j++) {
                float tmp = d.coordinates[j] - secret_layer[i].weights[j];
                secret_layer[i].output += tmp * tmp;
            }
            if (secret_layer[i].sigma != 0.f)
                secret_layer[i].output = exp(-sqrtf(secret_layer[i].output)/secret_layer[i].sigma);
            else
                secret_layer[i].output = 0.f;
        }
        // outputs of output layer and classification of d
        for (unsigned i = 0; i < output_layer.size(); i++) {
            output_layer[i].output = 0.f;
            for (unsigned j = 0; j < output_layer[i].weights.size(); j++) {
                output_layer[i].output += output_layer[i].weights[j] * secret_layer[j].output;
            }
            if (output_layer[i].output > output_layer[answer].output)
                answer = i;
        }
        correct += answer == d.cls;
    }

    return correct;
}


// simulate training of neurons
void train(vector<output_neuron_t> output_layer, vector<secret_neuron_t> secret_layer, vector<input_data> data, float mi)
{
    cout << "Nechť i a j jsou index neuronu výstupní vrstvy a index neuronu skryté vrstvy.\n";
    cout << "d je chtěná hodnota na výstupu neuronu i, y_i a y_j jsou hodnotu na výstupech neuronů i a j\n";
    cout << "i j : váha += mi * (d - y_j) * y_i = nová váha\n";
    cout << "Na vstup budou dokola přikládána vstupní data a váhy výstupní vrstvy neuronů budou přepočítávány dle vzorce výše.\n";
    cout << "Pod přepočty z jednoho vektoru dat bude informace o tom kolik dat by aktu´´alně bylo zařazeno správně s novými vahami.\n";
    cin.get();
    
    while (1) {
        for (unsigned i = 0; i < data.size(); i++) {
            input_data d = data[i];
            cout << "Vektor dat " << i << "/" << data.size() << ":";
            for (float cood: d.coordinates) {
                cout << " " << cood;
            }
            cout << endl;
            
            for (unsigned i = 0; i < secret_layer.size(); i++) {
                secret_layer[i].output = 0.f;
                for (unsigned j = 0; j < secret_layer[i].weights.size(); j++) {
                    float tmp = d.coordinates[j] - secret_layer[i].weights[j];
                    secret_layer[i].output += tmp * tmp;
                }
                if (secret_layer[i].sigma != 0.f)
                    secret_layer[i].output = exp(-sqrtf(secret_layer[i].output)/secret_layer[i].sigma);
                else
                    secret_layer[i].output = 0.f;
            }
            
            for (unsigned i = 0; i < output_layer.size(); i++) {
                output_layer[i].output = 0.f;
                for (unsigned j = 0; j < output_layer[i].weights.size(); j++) {
                    output_layer[i].output += output_layer[i].weights[j] * secret_layer[j].output;
                }

                for (unsigned j = 0; j < output_layer[i].weights.size(); j++) {
                    cout << i << " " << j << ": " << output_layer[i].weights[j] << " += " << mi << " * (" << (d.cls == i) << " - "
                        << output_layer[i].output << ") * " << secret_layer[j].output << " = " << output_layer[i].weights[j]
                            + mi * ((d.cls == i) - output_layer[i].output) * secret_layer[j].output << endl;
                    output_layer[i].weights[j] += mi * ((d.cls == i) - output_layer[i].output) * secret_layer[j].output;
                }

            }
            cout << "Správně: " << count_correct(output_layer, secret_layer, data) << "/" << data.size() << endl;
            cin.get();
        }
    }
}


int main(int argc, char *argv[])
{
    struct option long_opts[] =
    {
        {"help",       no_argument,        NULL, 'h'},
        {"random",     no_argument,        NULL, 'r'},
        {"class",      required_argument,  NULL, 'c'},
        {"dimension",  required_argument,  NULL, 'd'},
        {"file",       required_argument,  NULL, 'f'},
        {"mi",      required_argument,  NULL, 'm'},
        {"neuron",     required_argument,  NULL, 'n'},
        {0, 0, 0, 0}  // ukoncovaci prvek
    };

    float mi = 0.5f;
    const char *filename = "input";
    unsigned cls = 3, dim = 2, q1 = 6;
    bool rnd = false;
    srand(time(NULL));

    for (int c; (c = getopt_long(argc, argv, "hrc:d:f:m:n:", long_opts, nullptr)) != -1;) {
        switch (c) {
            case 'h':
                cout << "Přepínač --random/-r\n";
                cout << "\tPočáteční váhy neuronů skryté vrstvy budou vybrány náhodně z dat, bez přepínače budou použity prvnídata.\n";
                cout << "\tPočáteční váhy neuronů výstupní vrstvy budou náhodné v intervalu <-1;1>, bez přepínaše budou nulové.\n";
                cout << "Přepínač --class/-c X\n";
                cout << "\tX udává počet neuronů výstupní vrstvy, v základu 3.\n";
                cout << "Přepínač --dimension/-d X.\n";
                cout << "\tX udává dimenzi vstupních dat, v základu 2.\n";
                cout << "Přepínač --file/-f FILENAME\n";
                cout << "\tFILENAME udává jméno souboru, ze kterého se mají načítat vtupní data, v základu \"input\"\n.";
                cout << "Přepínač --mi/-m X\n";
                cout << "\tX udává koeficient učení výstupní vrstvy, v základu " << mi << ".\n";
                cout << "Přepínač --neuron/-n X\n";
                cout << "X udává počet neuronů skryté vrstvy, v základu 6.\n";
                return 0;
            case 'r':
                rnd = true;
                break;
            case 'c':
                cls = atoi(optarg);
                break;
            case 'd':
                dim = atoi(optarg);
                break;
            case 'f':
                filename = optarg;
                break;
            case 'm':
                if (float tmp; (tmp = atof(optarg)) != 0.f)
                    mi = tmp;
                else
                    cerr << "Koeficient učení nemůže být " << optarg << ", bude použit implicitní koeficient " << mi << endl;
                break;
            case 'n':
                q1 = atoi(optarg);
                break;
        }
    }

    vector<input_data> data = get_data(filename, dim, cls);
    if (data.size() < q1) {
        cerr << "Nedostatek vstupních dat pro inicializaci skryté vrstvy.\n";
        return EXIT_FAILURE;
    }

    cout << "Síť RBF " << dim << "/" << q1 << "/" << cls << endl;
    cout << "Koeficient učení: " << mi << endl;
    cout << "Inicializujeme " << q1 << " náhodných vektorů vah pro neurony skryté vrstvy s RBF.\n";
    if (rnd)
        cout << "Pro inicializaci budou použity náhodné (ale různé) vektory z množiny vstupů.\n";
    else
        cout << "Demonstrativně bude pro inicializaci použito " << q1 << " prvních vektorů z množiny vstupů.\n";

    cout << "Váhy:\n";

    vector<secret_neuron_t> secret_layer = init_secret_layer(data, q1, rnd);
    for (unsigned i = 0; i < secret_layer.size(); i++) {
        cout << "neuron " << i << ":";
        for (unsigned j = 0; j < secret_layer[i].weights.size(); j++) {
            cout << " " << fixed << setw(6) << setprecision(2) << secret_layer[i].weights[j];
        }
        cout << endl;
    }

    cin.get();

    cout << "Pro učení bude použit algoritmus K-means.\n";
    cout << "Nejprve se spočítá pro každý vstup trénovací množiny, kterému neuronu je nejblíže a uděláme z nich shluky.\n";
    cout << "Výpočet pro vstup i a váhy neuronu w: sqrt((i1-w1)^2 + (i2-w2)^2 + ... + (in-wn)^2)\n";
    cout << "Vstup | Vzdálenosti | minimální vzdálenost (číslo neuronu)\n";

    calculate_nearest_neurons(&data, secret_layer);
    cin.get();

    cout << "Následuje přepočtení vah neuronů tak, aby odpovídaly těžištím jejich shlukům.\n";
    cout << "K tomu stačí spočítat sumu vektorů ve shluku a podělit výsledek podělit počtem vektorů ve shluku.\n";

    calculate_center(&secret_layer, data);
    cout << "Nyní budeme opakovat výpočet nejbližších neuronů a jejich těžišť, dokud se těžiště neustálí.\n";
    cout << "(Tzn. po iteraci se nezmění pro žádný vstup jeho nejbližší neuron.)\n";
    cin.get();

    while (calculate_nearest_neurons(&data, secret_layer)) {
        cin.get();
        calculate_center(&secret_layer, data);
    }

    cout << "V tomto kroce již těžiště zůstala stejná, váhy neuronů skryté vrstvy jsou tak nastavené.\n";
    cout << "Následuje výpočet parametru sigma jako směrodatné odchylky vastupů náležících neuronu od jeho těžiště.\n";
    cout << "Množinu vektorů dat, které mají nejblíže daný neuron označnme jako C a mohutnost množiny jako n_c.\n";
    cout << "Vzorec pak vypadá jako sqrt(1/(n_c - 1) * suma(||i - w||^2))\n";
    cout << "kde w jsou váhy neuronu pro který počítáme a i jsou souřadnice bodů patřící do C.\n";

    cin.get();
    calculate_sigma(&secret_layer, data);
    cout << "Dále budou inicializovány vektory vah neuronů výstupní vrstvy.\n";
    if (rnd) {
        cout << "Vektory budou inicializovány náhodnými hodnotami v rozmezí <-1.0;1.0>\n";
    } else {
        cout << "Vektory budou demonstrativně inicializovány hodnotou 0.\n";
    }
    
    cin.get();
    train(init_output_layer(secret_layer.size(), cls, rnd), secret_layer, data, mi);

    return EXIT_SUCCESS;
}
