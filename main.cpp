// -*- C++ -*-
//===--------------------------- Q-Learning ---------------------------------===//
//
//                     C++ Version of Q-Learning
//
// This file implements an simple C++ Q-Learning algorithm
// Author is Jin Fagang.
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <fstream>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include "utils/utils.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

#define MAP_HEIGHT 100
#define SPACE_BETWEEN 100
#define PIPE 10
#define MAP_WIDTH 300

#define ACTION_NUM 2

#define MAX_EPISODE 10000
#define ALPHA 0.95

#define SHIFT_BIRD 10
#define JUMP 2
#define DOWN 1

double R[MAP_HEIGHT][SPACE_BETWEEN][MAP_HEIGHT][ACTION_NUM];
double Q[MAP_HEIGHT][SPACE_BETWEEN][MAP_HEIGHT][ACTION_NUM];

vector<pair<int, int>> obstacles;

bool cmp(const pair<int, int> &p1, const int v)
{
	if (p1.first<v)
		return true;
	else
		return false;
}

vector<pair<int, int>>::iterator getNextObstacle(int t) {
	return lower_bound(obstacles.begin(), obstacles.end(), t + SHIFT_BIRD, cmp);
}

void generateMap() {

	obstacles.clear();
	for (int i = SPACE_BETWEEN; i <= SPACE_BETWEEN * PIPE; i += SPACE_BETWEEN) {
		obstacles.push_back({ i, (rand() % (MAP_HEIGHT - JUMP * 2)) + JUMP });
	}
}

int playGame(int &position, int action, int t, bool draw = false) {

	// update status of bird
	if (action == 1)
		position -= JUMP - DOWN;
	else
		position += DOWN;

	auto obstacle = getNextObstacle(t);
//	cout << "time : " << t << " | bird: (" << SHIFT_BIRD << " , " << position << ") => next obstacle (" << (obstacle->first - t) << " , " << (obstacle->second) << ")" << endl;
	// draw game
	if (draw) {

		Mat game(MAP_HEIGHT, MAP_WIDTH, CV_8UC3, Scalar(100, 100, 100));

		auto temp = obstacle;

		while (temp->first - t < MAP_WIDTH) {
			for (int i = 0; i < MAP_HEIGHT; i++) {
				if (i - 9 <= temp->second && temp->second <= i + 9);
				else
					game.at<Vec3b>(i, temp->first - t) = Vec3b(0, 255, 0);
			}

			temp++;
		}

		for (int i = SHIFT_BIRD; i <= SHIFT_BIRD; i++) {
			for (int j = position - 5; j <= position + 5; j++) {
				if(i>=0 && i<= MAP_WIDTH && j>=0 && j<MAP_HEIGHT)
					game.at<Vec3b>(j, i) = Vec3b(0, 0, 255);
			}
		}
	

		String s = "g";
		resize(game, game, Size(MAP_WIDTH * 2, MAP_HEIGHT * 2));
		imshow(s, game);
	}

	// check status of game (win / lose / nothing)
	if (position <= 0 || position >= MAP_HEIGHT - 1)
		return 1;

	if (obstacle->first - t == SHIFT_BIRD) {
		if (abs(obstacle->second - position) > 2)
			return 1;
		else
			return 2;
	}
	
	return 0;
}


double get_max_q(int position, int action, int t, vector<pair<int,int>>::iterator obstacle){
    double temp_max = 0;

	int next_position;
	if (action == 0)
		next_position = position + DOWN;
	else
		next_position = position - JUMP + DOWN;

	for (int i = 0; i < ACTION_NUM; ++i) {
        if ((R[next_position][obstacle->first - t - 1][obstacle->second][i] >= 0) && (Q[next_position][obstacle->first - t - 1][obstacle->second][i] > temp_max)){
			temp_max = Q[next_position][obstacle->first - t - 1][obstacle->second][i];
        }
    }
    
	return temp_max;
}

void episode_iterator(int position){

	int action;
    double max_q;

    // start series event loop
    int t = 0;
    while (true){

		// get next action
		if (position <= JUMP * 2)
			action = 0;
		else if (position >= MAP_HEIGHT - JUMP * 2)
			action = 1;
		else
			action = rand() % 2;

		// treat next action as state, and we can get max{Q(s', a')}
		auto obstacle = getNextObstacle(t);
		max_q = get_max_q(position, action, t, obstacle);

        // update formula Q(s,a)=R(s,a)+alpha * max{Q(s', a')}
        Q[position][obstacle->first - t][obstacle->second][action] = R[position][obstacle->first - t][obstacle->second][action] + ALPHA * max_q;

		int result = playGame(position, action, t);
		if (result == 1) {
			break;
		}

		t++;
    }

}

int inference_best_action(int position, int t){
    // get the max value of Q corresponding action when state is nw_state
    double temp_max_q=0;
    int best_action=0;

	auto obstacle = getNextObstacle(t);

    for (int i = 0; i < ACTION_NUM; ++i) {
		cout << Q[position][obstacle->first - t][obstacle->second][i] << endl;
		cout << position << " " << obstacle->first - t << " " << obstacle->second << " " << i << endl;
        if (Q[position][obstacle->first - t][obstacle->second][i] > temp_max_q){
			temp_max_q = Q[position][obstacle->first - t][obstacle->second][i];
            best_action = i;
        }
    }
    return best_action;
}

void run_training(){

    // start random
    cout << "[INFO] start training..." << endl;
    for (int i = 0; i < MAX_EPISODE; ++i) {
		cout << "[INFO] Episode: " << i << endl;

		int position = rand() % MAP_HEIGHT;
		generateMap();

		episode_iterator(position);

    }
}

void saveQMatrix() {
	FILE *fp = fopen("./../QMatrix.txt", "w");
	for (int i = 0; i < MAP_HEIGHT; i++) {
		for (int j = 0; j < SPACE_BETWEEN; j++) {
			for (int k = 0; k < MAP_HEIGHT; k++) {
				fprintf(fp, "%lf %lf\n", Q[i][j][k][0], Q[i][j][k][1]);
			}
		}
	}
	fclose(fp);
}

void loadQMatrix() {
	FILE *fp = fopen("./../QMatrix.txt", "r");
	for (int i = 0; i < MAP_HEIGHT; i++) {
		for (int j = 0; j < SPACE_BETWEEN; j++) {
			for (int k = 0; k < MAP_HEIGHT; k++) {
				fscanf(fp, "%lf %lf", &Q[i][j][k][0], &Q[i][j][k][1]);
			}
		}
	}
	fclose(fp);
}


void initialRMatrix() {
	for (int i = 0; i < MAP_HEIGHT; i++) {
		for (int j = 0; j < SPACE_BETWEEN; j++) {
			for (int k = 0; k < MAP_HEIGHT; k++) {
				if (i <= JUMP * 2)
					R[i][j][k][1] = -10000000;
				if (i > MAP_HEIGHT - JUMP * 2)
					R[i][j][k][0] = -10000000;

				if (j == SHIFT_BIRD - 1 && k == i - JUMP)
					R[i][j][k][1] = 10000000;
				if (j == SHIFT_BIRD - 1 && i + DOWN == k)
					R[i][j][k][0] = 10000000;
			}
		}
	}
}

int main() {
	srand((unsigned)time(NULL));

	cout << "initial" << endl;
	initialRMatrix();
	loadQMatrix();

	int counter = 0;
	//remove this while for real test
	while (true) {
		counter++;
		cout << "train" << counter << endl;
		run_training();
		saveQMatrix();
		cout << "saved" << endl;
	}

	
	namedWindow("temp");
	while (true){

		int position = rand() % MAP_HEIGHT;
		generateMap();
		int t = 0;

		while (true){
            int best_action = inference_best_action(position, t);

			int result = playGame(position, best_action, t, true);

			if (result == 1) {
				cout << "GG";
				break;
			}

			t++;
			waitKey(50);
        }
    }
	

    return 0;
}