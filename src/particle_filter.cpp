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
	default_random_engine gen;
	num_particles=100;

	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	for(int i=0; i<num_particles; i++){
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight =1;
		particles.push_back(p);
		weights.push_back(1);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;
	for(int i=0; i<num_particles; i++){
		double x = particles[i].x;
		double y = particles[i].y;
		double theta = particles[i].theta;
		double theta_final = theta + yaw_rate*delta_t;
		if(yaw_rate!=0){
			x = x + (velocity/yaw_rate)*(sin(theta_final)-sin(theta));
			y = y + (velocity/yaw_rate)*(cos(theta)-cos(theta_final));
		}else{
			x = x + (velocity)*delta_t*cos(theta);
			y = y + (velocity)*delta_t*sin(theta);
		}

		theta = theta_final;

		normal_distribution<double> dist_x(x, std_pos[0]);
		normal_distribution<double> dist_y(y, std_pos[1]);
		normal_distribution<double> dist_theta(theta, std_pos[2]);

		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for(int i=0; i<observations.size();i++){
		double min_distance;
		int nearest_landmark_id=-1;
		for(int j=0;j<predicted.size();j++){
			double cur_distance = dist(observations[i].x, observations[i].y, predicted[j].x,predicted[j].y);
			if(nearest_landmark_id==-1 || cur_distance<min_distance){
				min_distance = cur_distance;
				nearest_landmark_id = predicted[j].id;
			}
		}
		observations[i].id = nearest_landmark_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
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
	for (int i=0;i<num_particles;i++)
	{
		vector<LandmarkObs> predicted;
		//Compute the map landmarks which are within the sensor range
		for (int j=0;j<map_landmarks.landmark_list.size();j++)
		{
			Map::single_landmark_s landmark = map_landmarks.landmark_list[j];
			double distance=dist(landmark.x_f,landmark.y_f,particles[i].x,particles[i].y);
			if (distance<sensor_range)
			{
				LandmarkObs obs;
				obs.id = landmark.id_i;
				obs.x = landmark.x_f;
				obs.y = landmark.y_f;
				predicted.push_back(obs);
			}
		}

		vector<LandmarkObs> transformed_observations;

		//Transform the observations from vehicle co-ordinate system to map co-ordinate system.
		for (int j=0;j<observations.size();j++)
		{
			LandmarkObs trans_obs;
			trans_obs.x=particles[i].x+cos(particles[i].theta)*observations[j].x-sin(particles[i].theta)*observations[j].y;
			trans_obs.y=particles[i].y+sin(particles[i].theta)*observations[j].x+cos(particles[i].theta)*observations[j].y;
			transformed_observations.push_back(trans_obs);
		}

		//Associate each observation with a landmark
		dataAssociation(predicted, transformed_observations);

		double particle_weight = 1.0;
		//Compute the new weight for particle using Multi-variate gaussian distribution
		for (int j=0;j<transformed_observations.size();j++)
		{
			LandmarkObs nearest_landmark;
			for (int k=0;k<predicted.size();k++)
			{
				if (predicted[k].id==transformed_observations[j].id)
				{
					nearest_landmark.id=predicted[k].id;
					nearest_landmark.x=predicted[k].x;
					nearest_landmark.y=predicted[k].y;
				}
			}
			particle_weight*= multivariate_gaussian(transformed_observations[j].x, transformed_observations[j].y,nearest_landmark.x, nearest_landmark.y, std_landmark[0], std_landmark[1] );
		}
		particles[i].weight = weights[i] = particle_weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine gen;
	discrete_distribution<int> distribution(weights.begin(), weights.end());
	vector<Particle> temp;
	for(int i=0;i<num_particles;i++){
		temp.push_back(particles[distribution(gen)]);
	}
	particles =temp;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
		const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	particle.associations= associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;
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
