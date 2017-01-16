#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <list>
#include <memory>
#include <string>
#include <algorithm>
#include <unordered_set>

/*
	
	This is a problem of multilabel classification where we have 
	text as an input and multiple labels as an output, each text may
	be occured in more then one category. 

	To solve this problem we need following steps. 
	1) Convert text into vectors, with tfidf method, it's much 
		better rathen then just bag of word representation 
		in order to have much better model we should include ngrams
		while converting text into a some vector space model 

	2) after this we can use any classification model like 
		multinomial naive bayes, logistic rergession and so on, 
		but I think best one in this case in general would be 
		logistic regression. 

	3) We should do something like binary relevance approach for multilabel 
		classification. 

	
	Bellow is not an implementation of this approach, I'm also sending python script 
	which uses sklearn library and doing all of this I've declared above. 

	Bellow is naive bayes approach we I'm count frequencies of wors in each category 
	after this I'm computing probabilities for all classes and picking up 
	the best ones. 

	I've implemented this way becouse it would be very hard to implemented everything from 
	scratch in c++ (tfidf, logistic regression and binary relevance method). 

	In python implementation there is nothing impressive I think because I'm just using 
	already implemneted and tested libraries. 

	It was so great experience while working on this challenge I've learnt so many things. 

	Naive bayes implementation is based on open source github project which is written is java. 

*/


/*
	This is a wrapper class which get 
	Cateogyry and array of Features 
	it's used in classifier. 
*/	

template<typename Category, typename Feature> 
class Classification
{
 protected:
 	// The classified featurreset
 	std::vector<Feature> feature_set_;


 	// The category as the feature_Set was classified 
 	Category category_;

 	// The probability that the featureset belongs to the given category. 
 	float probability_;
 public:
 	Classification();
 	~Classification();

 	Classification(const std::vector<Feature>& featureset, Category category);

 	Classification(const std::vector<Feature>& featureset, Category category, 
 				   float probability);



 	std::vector<Feature> get_featureset() const;

 	/* returns probability, which after this used for sorting classifications */
 	float get_probability() const;

 	/* retursn categy of Classification */
 	Category get_category() const;


 	/* thisis operator for comparing classifications, it compares classifiers base
 	   on probability it's used in classifier class to sort categorie s
 	   according to their probabilities, and pick up the classes 
 	   with highest probabilities */
 	bool operator<(const Classification<Category, Feature>& other) {
 		return this->get_probability() < other.get_probability();
 	}


};



template<typename Category, typename Feature> 
Classification<Category, Feature>::Classification() {
	// Constructor of classification 
	// There is nothing to do
}

template<typename Category, typename Feature> 
Classification<Category,Feature>::~Classification() 
{
	// 
}


template<typename Category, typename Feature>
Classification<Category, Feature>::Classification(const std::vector<Feature>& featureset, 
												  Category category)
	: Classification(featureset, category, 1.0)
{
	//this->Classification(featureset, category, 1.0);
}


template<typename Category, typename Feature>
Classification<Category, Feature>::Classification(const std::vector<Feature>& featureset, 
												  Category category,
												  float probability)
	: feature_set_(featureset)
	, category_(category)
	, probability_(probability)
{

}



template<typename Category, typename Feature>
std::vector<Feature> Classification<Category, Feature>::get_featureset() const {
	return this->feature_set_;
}

template<typename Category, typename Feature> 
Category Classification<Category, Feature>::get_category() const {
	return this->category_;
}

template<typename Category, typename Feature>
float Classification<Category, Feature>::get_probability() const {
	return this->probability_;
}



// Interface for computing probabilities
// if there is any other way to compute probabilitie 
// of (feature, category) class you should implement this 
// interface 
// we aren't going to use this for our implementation 
// but it's for general case
template<typename Category, typename Feature> 
class IFeatureProbability
{
 public:
 	virtual float feature_probability(Feature feature, Category category) = 0;
};



template<typename Category, typename Feature> 
class Classifier
{
 protected:

 	/* A dictionary mapping features to their number of occurences in each 
 		known category */
 	std::unordered_map<Feature, std::unordered_map<Category, int>> feature_count_per_category_;

 	/* 
 	  A dictionary mapping features to their number of occurences
 	*/

 	std::unordered_map<Category, int> total_category_count_;

 	/* A dictionary mapping to features to their number of occurenses */
 	std::unordered_map<Feature, int> total_feature_count_;

 public:
 	Classifier();
 	~Classifier();

 	/* Returns known features */

 	std::vector<Feature> get_features() const;
 	
 	/* Returns know categories */
 	std::vector<Category> get_categories() const;

 	/* returns the total number of categories */
 	int get_categories_total();

 	/* Increments the count of a given feature in the given category */

 	void increment_feature(Feature feature, Category category);

 	/* Increments the count of a given category */
 	void increment_category(Category category);

 	/* Decrements the coutn of a given feature in the given category */
 	void decrement_feature(Feature feature, Category category);

 	/* Decrements the count of a given category */
 	void decrement_category(Category category);

 	/* Retrives the number of occurences of the given feature in the 
 	given category */
 	int feature_count(Feature feature, Category category);

 	/* Retrives the number of occurences of the given category */
 	int category_count(Category category);

 	/* computes the probaiblity of (feature, category) */
 	float feature_probability(Feature feature, Category category);

 	/* this functions used to learn classifier */
 	void learn(Category category, const std::vector<Feature>& features);

 	/* There we use Classification class for wraping Category and features */
 	void learn(Classification<Category, Feature> classification);


 	float feature_weighted_average(Feature feature, Category category);

 	float feature_weighted_average(Feature feature, Category category, 
 								   IFeatureProbability<Feature, Category>* calculator);


 	float feature_weighted_average(Feature feature, Category category,
 								   IFeatureProbability<Feature, Category>* calculator,
 								   float weight);

 	float feature_weighted_average(Feature feature, Category category,
 							       IFeatureProbability<Feature, Category>* calculator,
 							       float weight, float assumed_probability);

 	void build_useless(int k = 50);

 	std::unordered_set<Feature> get_useless() const;

 	bool is_useless(const Feature& feature) const;


};

template<typename Category, typename Feature>
Classifier<Category, Feature>::Classifier() {

}

template<typename Category, typename Feature>
Classifier<Category, Feature>::~Classifier() {

}

template<typename Category, typename Feature>
std::vector<Feature> Classifier<Category, Feature>::get_features() const {
	std::vector<Feature> result;
	result.reserve(total_feature_count_.size());
	for(auto it = total_feature_count_.begin(); it != total_feature_count_.end(); ++it) {
		result.push_back(it->first);
	}
	return result;
}


template<typename Category, typename Feature>
std::vector<Category> Classifier<Category, Feature>::get_categories() const {
	std::vector<Category> result;
	result.reserve(total_category_count_.size());

	for(auto it = total_category_count_.begin(); it != total_category_count_.end(); ++it) {
		result.push_back(it->first);
	}
	return result;
}


template<typename Category,typename Feature>
int Classifier<Category, Feature>::get_categories_total() {
	int result = 0;
	for(auto it = total_category_count_.begin(); it != total_category_count_.end(); ++it) {
		result += it->second;
	}
	return result;
}

template<typename Category, typename Feature>
void Classifier<Category, Feature>::increment_feature(Feature feature, Category category) {
	if(feature_count_per_category_.find(category) == feature_count_per_category_.end()) {
		std::unordered_map<Feature, int> features;
		features[feature] = 1;
		feature_count_per_category_[category] = features;
	} else {
		feature_count_per_category_[category][feature] += 1;
	}

	total_feature_count_[feature] += 1;
}


template<typename Category, typename Feature>
void Classifier<Category, Feature>::increment_category(Category category) {
	this->total_category_count_[category] += 1;
}

template<typename Category, typename Feature>
int Classifier<Category, Feature>::feature_count(Feature feature, Category category) {
	if(feature_count_per_category_.find(category) == feature_count_per_category_.end()) {
		return 0;
	} else {
		return feature_count_per_category_[category][feature];
	}
}

template<typename Category, typename Feature> 
int Classifier<Category, Feature>::category_count(Category category) {
	return total_category_count_[category];
}

#include <cassert>



template<typename Category, typename Feature> 
float Classifier<Category, Feature>::feature_probability(Feature feature, Category category) {
	if(this->category_count(category) == 0) {
		return 0;
	}
	if(is_useless(feature)) {
		return 1.0f;
	}


	

	return (float) this->feature_count(feature, category) / (float) this->category_count(category);
}


template<typename Category, typename Feature>
void Classifier<Category, Feature>::learn(Category category, const std::vector<Feature>& features) {
		this->learn(Classification<Category, Feature>(features, category));
}

template<typename Category, typename Feature>
void Classifier<Category, Feature>::learn(Classification<Category, Feature> classification) {
	for(auto feature : classification.get_featureset()) {
		if(is_useless(feature)) {
			continue;
		}
		this->increment_feature(feature, classification.get_category());
	}
	this->increment_category(classification.get_category());

}


template<typename Category, typename Feature>
float Classifier<Category, Feature>::feature_weighted_average(Feature feature, Category category) {
	return this->feature_weighted_average(feature, category, nullptr, 1.0f, 0.5f);
}

template<typename Category, typename Feature>
float Classifier<Category, Feature>::feature_weighted_average(Feature feature, Category category,
															  IFeatureProbability<Feature, Category>* calculator) {
	return this->feature_weighted_average(feature, category, calculator, 1.0f, 0.5f);
}


template<typename Category, typename Feature>
float Classifier<Category, Feature>::feature_weighted_average(Feature feature, Category category,
															  IFeatureProbability<Feature, Category>* calculator,
															  float weight) {
	return this->feature_weighted_average(feature, category, calculator, weight, 0.5f);
}

template<typename Category, typename Feature>
float Classifier<Category, Feature>::feature_weighted_average(Feature feature, Category category,
															  IFeatureProbability<Feature, Category>* calculator,
															  float weight, 
															  float assumed_probability) {
	float basic_probability = (calculator == nullptr) ? this->feature_probability(feature, category)
		: calculator->feature_probability(feature, category);

	int totals = total_feature_count_[feature];

	return (weight * assumed_probability + totals * basic_probability) 
				/ (weight + totals);


}


template<typename Category,typename Feature>
class BayesClassifier : public Classifier<Category, Feature>
{

 public:
 	float feature_probability(std::vector<Feature> features, 
 							  Category category);

 	float category_probability(std::vector<Feature> features, 
 		                       Category category);

 	std::vector<Classification<Category, Feature>> category_probabilities(std::vector<Feature> features);

 	Classification<Category, Feature> classify(std::vector<Feature> features);

 	std::vector<Classification<Category, Feature>> calssify_detailed(std::vector<Feature> features);

 	float average_probability(const std::vector<Classification<Category, Feature>>& classifications);



};

template<typename Category, typename Feature> 
float BayesClassifier<Category, Feature>::average_probability(const std::vector<Classification<Category, Feature>>& classifications) {
	float sum = 0;

	for(int i = 0; i < classifications.size(); ++i) {
		sum += classifications[i].get_probability();
	}

	return sum / classifications.size();
}

template<typename Category, typename Feature>
float BayesClassifier<Category, Feature>::feature_probability(std::vector<Feature> features, 
														      Category category) {
	float product = 1.0f;

	for(Feature feature : features) {
		if(this->is_useless(feature)) {
			continue;
		}
		product *= this->feature_weighted_average(feature, category);
	}

	return product;
}



template<typename Category, typename Feature>
float BayesClassifier<Category, Feature>::category_probability(std::vector<Feature> features,
															   Category category) {
	return ( (float) this->category_count(category) / (float) this->get_categories_total()) 
				* feature_probability(features, category);
}





template<typename Category, typename Feature>
std::vector<Classification<Category, Feature>> BayesClassifier<Category, Feature>::category_probabilities(
	std::vector<Feature> features)
 {

	std::vector<Classification<Category, Feature>> classifications;

	for(Category category : this->get_categories()) {
		Classification<Category, Feature> classification(features, category, this->category_probability(features, category));
		classifications.push_back(classification);
	}

	float average = this->average_probability(classifications);

	std::sort(classifications.begin(), classifications.end());
	std::reverse(classifications.begin(), classifications.end());

	return classifications;
}


template<typename Category, typename Feature>
Classification<Category, Feature> BayesClassifier<Category, Feature>::classify(std::vector<Feature> features) {

	std::vector<Classification<Category,Feature>> probabilities = this->category_probabilities(features);

	if(probabilities.size() > 0) {
		return probabilities[probabilities.size()-1];
	}

	//return nullptr;
}

template<typename Category, typename Feature>
std::vector<Classification<Category, Feature>> BayesClassifier<Category,Feature>::calssify_detailed(
	std::vector<Feature> features) {
	return this->category_probabilities(features);
}



#include <set>

template<typename Category, typename Feature>
void Classifier<Category, Feature>::build_useless(int k) {

	std::set<std::pair<int, Feature>> table;

	for(auto it = this->total_feature_count_.begin(); it != this->total_feature_count_.end(); ++it) {
		std::pair<int, Feature> p = std::make_pair(it->second, it->first);

		if(table.size() < k) {
			table.insert(p);
		} else {
			auto set_iterator = table.begin();
			std::pair<int, Feature> smallest_pair = *set_iterator;

			if(smallest_pair.first < p.first) {
				table.erase(table.begin());
				table.insert(p);
			}

		}
	}

	for(auto it = table.begin(); it != table.end(); ++it) {
		useless_.insert(it->second);
	}

}

template<typename Category, typename Feature>
std::unordered_set<Feature> Classifier<Category, Feature>::get_useless() const {
	return this->useless_;
}

template<typename Category, typename Feature>
bool Classifier<Category, Feature>::is_useless(const Feature& feature) const {
	if(this->useless_.find(feature) == this->useless_.end()) 
		return false;
	return true;
}




#include <sstream>

std::vector<std::string> create_vector(const std::string& line) {
	std::vector<std::string> vec;
	std::stringstream ss;
	ss << line;
	std::string word;
	while(ss >> word) {
		if(word[word.size()-1] == '?') {
			vec.push_back(word.substr(0, word.size()-1));
		} else {
			vec.push_back(word);
		}
		
	}
	return vec;
}

int main() {
	int T, E;
	BayesClassifier<std::string, std::string> bayes;

	if(std::cin >> T >> E) {
		std::cin.ignore();
	}
		
	
	//std::unordered_map<std::string, int> category_table;

	for(int i = 0; i < T; ++i) {
		std::string line_1;
		std::string line_2;

		std::getline(std::cin, line_1);
		std::getline(std::cin, line_2);

		auto vec_1 = create_vector(line_1);
		auto vec_2 = create_vector(line_2);



		for(int i = 0; i < vec_1.size(); ++i) {
			//category_table[vec_1[i]] += 1;
			bayes.learn(vec_1[i], vec_2);
		}

	}

	
	bayes.build_useless(50);




	
	for(int i = 0; i < E; ++i) {
		std::string line;
		std::getline(std::cin, line);
		std::vector<std::string> vec = create_vector(line);

		int counter = 0;

		for(int i = bayes.calssify_detailed(vec).size() - 1; i >= bayes.calssify_detailed(vec).size() - 10; i--) {
			std::cout << bayes.calssify_detailed(vec)[i].get_category() << " ";
		} std::cout << std::endl;

	}

		

	




		










	return 0;
	
}