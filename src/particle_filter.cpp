/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 200;

	default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for(int i =0; i<num_particles; i++) {
		Particle particle;
		particle.id = i;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		particle.weight = 1.0;
		particles.push_back(particle);
		weights.push_back(1.0);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;

	for (Particle &particle: particles) {

		double new_x;
		double new_y;
		double new_theta;

		if(fabs(yaw_rate) == 0) {
			new_x =  particle.x + velocity*delta_t*cos(particle.theta);
			new_y = particle.y + velocity*delta_t*sin(particle.theta);
			new_theta = particle.theta;

		}
		else {
			new_x = particle.x + (velocity / yaw_rate) * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta)) ;
			new_y = particle.y + (velocity / yaw_rate) * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t)) ;
			new_theta = particle.theta + yaw_rate * delta_t;
		}

		normal_distribution<double> noise_x(new_x, std_pos[0]);
		normal_distribution<double> noise_y(new_y, std_pos[1]);
		normal_distribution<double> noise_theta(new_theta, std_pos[2]);

		particle.x =  noise_x(gen);
		particle.y =  noise_y(gen);
		particle.theta = noise_theta(gen);

	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	double sig_x = pow(std_landmark[0], 2);
	double sig_y = pow(std_landmark[1], 2);
	double sig_xy = std_landmark[0]*std_landmark[1];
	for (int i = 0; i < num_particles; i++) {
		Particle& particle = particles[i];
		double weight = 1.0;

		for (int j=0; j < observations.size(); j++) {
			LandmarkObs obs = observations[j];
			double predicted_x = obs.x *cos(particle.theta) - obs.y*sin(particle.theta) + particle.x;
			double predicted_y = obs.x *sin(particle.theta) + obs.y*cos(particle.theta) + particle.y;

			Map::single_landmark_s nl;
			double min_distance = sensor_range;
			double distance = 0;
			for (int k = 0; k<map_landmarks.landmark_list.size(); k++) {
				Map::single_landmark_s sl = map_landmarks.landmark_list[k];
				distance = sqrt(pow((sl.x_f - predicted_x),2) + pow((sl.y_f - predicted_y),2));
				if(distance < min_distance)  {
					min_distance = distance;
					nl = sl;
				}
			}
			// calculate normalization term
			double gauss_norm= ((2 * M_PI * sig_xy));

			// calculate exponent
			double exponent= (pow((predicted_x - nl.x_f),2))/(sig_x) +
					(pow((predicted_y - nl.y_f),2))/(sig_y);

			// calculate weight using normalization terms and exponent
			double normw =   exp(-0.5*exponent)/gauss_norm;
			weight *= normw;
		}
		cout << "weight: " << weight << endl;
		particle.weight = weight;
		weights[i] = weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine gen;
	discrete_distribution<> d(weights.begin(),weights.end());
	vector<Particle> new_samples;
	for (int i = 0; i<num_particles; i++) {
		new_samples.push_back(particles[d(gen)]);
	}
	particles = new_samples;

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
