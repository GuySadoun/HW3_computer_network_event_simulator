import numpy as np
import pandas as pd
import sys
from scipy.stats import poisson

#global
TOTAL_TIME, NUM_OF_VACCINES, LAMBDA_ARRIVAL, MU_SERVICE, PROBABILITIES, QUEUE_LEN = 0, 0, 0, 0, [], 0

def gen_int_arr():  # function to generate arrival times using inverse trnasform
    return -np.log((np.random.uniform(low=0.0, high=1.0))) / LAMBDA_ARRIVAL


def gen_service_time_teller1():  # function to generate service time for teller 1 using inverse trnasform
    return -np.log((np.random.uniform(low=0.0, high=1.0))) / MU_SERVICE


def gen_service_time_teller2():  # function to generate service time for teller 1 using inverse trnasform
    return -np.log((np.random.uniform(low=0.0, high=1.0))) / MU_SERVICE


def choose_vaccine():
    return np.random.randint(low=0, high=NUM_OF_VACCINES - 1)


def gen_prob():  # function to generate service time for teller 1 using inverse trnasform
    return np.random.uniform(low=0.0, high=1.0)


class Simulation:
    def __init__(self):
        self.clock = 0.0  # simulation clock
        self.num_arrivals = 0  # total number of arrivals
        self.t_arrival = gen_int_arr()  # time of next arrival
        self.t_departure = float('inf')  # departure time from server
        self.num_in_q = 0  # current number in queue
        self.states = np.zeros(shape=NUM_OF_VACCINES)  # current state of servers (binary)
        self.current_q_id = choose_vaccine()  # vaccine chosen by costumers/current queue

        self.dep_sum = np.zeros(shape=NUM_OF_VACCINES)  # Sum of service times
        self.num_of_departures = np.zeros(shape=NUM_OF_VACCINES)  # number of customers served
        self.total_wait_time = 0.0  # total wait time
        self.waited_in_queue = 0  # customers who had to wait in line(counter)
        self.lost_customers = 0  # customers who left without service

        # self.t_departure1 = float('inf')  # departure time from server 1
        # self.t_departure2 = float('inf')  # departure time from server 2
        # self.dep_sum1 = 0  # Sum of service times by teller 1
        # self.dep_sum2 = 0  # Sum of service times by teller 2
        # self.state_T1 = 0  # current state of server1 (binary)
        # self.state_T2 = 0  # current state of server2 (binary)
        # self.num_of_departures1 = 0  # number of customers served by teller 1
        # self.num_of_departures2 = 0

    def time_adv(self):
        t_next_event = min(self.t_arrival, self.t_departure)
        self.total_wait_time += (self.num_in_q * (t_next_event - self.clock))
        self.clock = t_next_event

        if self.t_arrival < self.t_departure:
            self.arrival()
        else:
            self.depart()

    def arrival(self):
        self.num_arrivals += 1

        if self.num_in_q == 0:  # schedule next departure or arrival depending on state of servers
            self.current_q_id = choose_vaccine()
            if self.states[self.current_q_id] == 0:
                self.states[self.current_q_id] = 1
                t_service = gen_service_time_teller1()
                self.dep_sum[self.current_q_id] += t_service
                self.t_departure = self.clock + t_service
                self.t_arrival = self.clock + gen_int_arr()
            else:
                self.num_in_q += 1
                self.waited_in_queue += 1
                self.t_arrival = self.clock + gen_int_arr()
        elif 1 <= self.num_in_q < QUEUE_LEN:
            p = gen_prob()
            if p <= PROBABILITIES[self.num_in_q]:
                print(self.num_in_q)
                self.num_in_q += 1
                self.waited_in_queue += 1
                self.t_arrival = self.clock + gen_int_arr()
            else:
                self.lost_customers += 1
        else:
            self.lost_customers += 1

    def depart(self):  # departure from server 2
        self.num_of_departures[self.current_q_id] += 1
        if self.num_in_q > 0:
            t_service = gen_service_time_teller1()
            self.dep_sum[self.current_q_id] += t_service
            self.t_departure = self.clock + t_service
            self.num_in_q -= 1
        else:
            self.t_departure = float('inf')
            self.states[self.current_q_id] = 0


def main():
    args = sys.argv[1:]
    global TOTAL_TIME, NUM_OF_VACCINES, LAMBDA_ARRIVAL, MU_SERVICE, PROBABILITIES, QUEUE_LEN
    TOTAL_TIME = int(args[0])
    NUM_OF_VACCINES = int(args[1])
    LAMBDA_ARRIVAL = int(args[2])
    MU_SERVICE = int(args[3])
    PROBABILITIES = np.array(args[4:], dtype=float)
    QUEUE_LEN = len(PROBABILITIES)
    print(
        f'Time: {TOTAL_TIME},Num of vaccuines: {NUM_OF_VACCINES}, Lambda: {LAMBDA_ARRIVAL}, Mu: {MU_SERVICE}, Probs: {PROBABILITIES}, queue len: {QUEUE_LEN}')  # Press Ctrl+F8 to toggle the breakpoint.
    s = Simulation()
    s.__init__()
    while s.clock <= TOTAL_TIME:
        s.time_adv()
    a = pd.Series([s.clock / s.num_arrivals, s.waited_in_queue, s.total_wait_time, s.lost_customers])
    print(a)
    #     df = df.append(a, ignore_index=True)
    # df = pd.DataFrame(
    #     columns=['Average interarrival time', 'Average service time teller1', 'Average service time teller 2',
    #              'Utilization teller 1', 'Utilization teller 2', 'People who had to wait in line',
    #              'Total average wait time', 'Lost Customers'])
    #
    # for i in range(100):
    #     np.random.seed(i)
    #     s.__init__()
    #     while s.clock <= 240:
    #         s.time_adv()
    #     a = pd.Series([s.clock / s.num_arrivals, s.dep_sum1 / s.num_of_departures1, s.dep_sum2 / s.num_of_departures2,
    #                    s.dep_sum1 / s.clock, s.dep_sum2 / s.clock, s.number_in_queue, s.total_wait_time,
    #                    s.lost_customers], index=df.columns)
    #     df = df.append(a, ignore_index=True)
    #
    # df.to_excel('results.xlsx')
    # LAMBDA_ARRIVAL = 1 / float(args.lambdaArrival)
    # MU_SERVICE = 1 / float(args.muService)
    # TOTAL_ARRIVALS = int(args.totalArrivals)
    #
    # sim = QueueSimulator(TOTAL_ARRIVALS, LAMBDA_ARRIVAL, MU_SERVICE)
    # sim.run()


if __name__ == '__main__':
    main()
