import sys
import numpy as np
import random as rand


class Simulation:
    def __init__(self, T, M, Lambda, Mu, probs):
        self.total_time = T
        self.num_of_vaccines = M
        self.lambda_arrival = Lambda
        self.mu_service = Mu
        self.probabilities = probs
        self.q_len = len(probs)

        self.clock = 0.0  # simulation clock
        self.customers_in_q = 0  # customers in queue - First in q gets service
        self.num_arrivals = 0  # total number of arrivals
        self.t_arrival = self.gen_next_arrival()  # time of next arrival
        self.t_departure = float('inf')  # time of next departure
        self.queues_len_time = np.zeros(shape=self.q_len, dtype=float)  # current state of each station
        self.current_q_id = self.choose_vaccine()  # vaccine chosen by costumers/current queue
        self.lost_customers = 0  # customers who left without service
        self.t_last_vaccine = 0  # time of last vaccine given

    def simulate(self):
        while self.clock <= self.total_time:
            self.time_adv()
        self.print_result()

    def time_adv(self):
        t_next_event = min(self.t_arrival, self.t_departure)
        t_waiting = (t_next_event - self.clock)
        self.clock = t_next_event
        self.queues_len_time[self.customers_in_q] += t_waiting
        if self.clock == self.t_arrival:
            self.arrival()
        else:
            self.depart()

    def arrival(self):
        self.num_arrivals += 1
        self.t_arrival += self.gen_next_arrival()
        if self.customers_in_q == 0:  # no one is waiting or getting service
            self.current_q_id = self.choose_vaccine()
            self.customers_in_q += 1
            self.t_departure = self.clock + self.gen_service_time()
        elif self.customers_in_q < self.q_len:
            assert self.customers_in_q > 0
            prob = np.random.uniform(low=0.0, high=1.0)
            if prob < self.probabilities[self.customers_in_q]:
                self.customers_in_q += 1
            else:
                self.lost_customers += 1

    def depart(self):
        self.t_last_vaccine = self.clock
        self.customers_in_q -= 1
        if self.customers_in_q > 0:
            self.t_departure = self.clock + self.gen_service_time()
        else:
            self.t_departure = float('inf')

    def print_result(self):
        served = self.num_arrivals - self.lost_customers
        total_waiting_time = sum((self.queues_len_time[x]*(x-1)) for x in range(2, self.q_len))
        total_service_time = sum(self.queues_len_time[1:])
        total_time = sum(self.queues_len_time)
        X, Y, T, T0 = served, self.lost_customers, self.t_last_vaccine, self.queues_len_time[0]
        T_w = total_waiting_time / served
        T_s = total_service_time / served
        lambda_A = served / total_time
        print(f'{X} {Y} {"%.2f" % T} {"%.3f" % self.queues_len_time[0]}', end=' ')
        for i in range(1, self.q_len):  # print A_Ti's including A_T0
            print(f'{"%.3f" % (self.queues_len_time[i] / self.num_of_vaccines)}', end=' ')
        print(f'{"%.6f" % (self.queues_len_time[0] / total_time)}', end=' ')
        for i in range(1, self.q_len):  # print Z_Ti's
            print(f'{"%.6f" % ((self.queues_len_time[i] / total_time) / self.num_of_vaccines)}', end=' ')
        print(f'{"%.7f" % T_w} {"%.7f" % T_s} {"%.2f" % lambda_A}')
        # print(f'X: {X} Y: {Y} T\': {"%.2f" % T}', end=' ')
        # print(f'T_0: {"%.3f" % self.queues_len_time[0]}', end=' ')
        # for i in range(1, self.q_len):  # print A_Ti's including A_T0
        #     print(f'T_{i}: {"%.3f" % (self.queues_len_time[i]/self.num_of_vaccines)}', end=' ')
        # print(f'Z_0: {"%.6f" % (self.queues_len_time[0]/total_time)}', end=' ')
        # for i in range(1, self.q_len):  # print Z_Ti's
        #     print(f'Z_{i}: {"%.6f" % ((self.queues_len_time[i]/total_time)/self.num_of_vaccines)}', end=' ')
        # print(f'T_w: {"%.7f" % T_w}', end=' ')
        # print(f'T_s: {"%.7f" % T_s}', end=' ')
        # print(f'Lambda_mean: {"%.2f" % lambda_A}')

    def gen_next_arrival(self):  # function to generate arrival times using inverse transform
        return rand.expovariate(self.lambda_arrival)

    def gen_service_time(self):
        return rand.expovariate(self.mu_service)

    def choose_vaccine(self):
        return np.random.randint(low=0, high=self.num_of_vaccines)


def main():
    # args = sys.argv[1:]
    args = [1000, 2, 60, 30, 1, 0.8, 0.5, 0]

    s = Simulation(float(args[0]), int(args[1]), float(args[2]), float(args[3]), np.array(args[4:], dtype=float))
    s.simulate()


if __name__ == '__main__':
    main()
