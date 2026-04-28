import numpy as np
import pickle
import time
import os
import matplotlib.pylab as plt
import chess_env.chess_game as bfun
import chess_env.chess_interface as bgra

# Q-learning strategy using dictionary for Q-values in each known state: 
class Strategy_Qdict:
    def __init__(self,game_obj):
        self.game_obj = game_obj          # game object - chess
        self.Q_dict = {}                  # dictionary - key: state, value: Q-table for al possible actions
        self.if_mixed = False             # if mixed strategy -> chosen from probability distribution
        self.distrib_dict = {}            # dictionary - key: state, value: cumulated sums of action probablities
                                          # to quick choose random action for mixed strategy 

    # general methods:

    def choose_action(self,state,player):
        key = self.game_obj.state_key(state)    # making key for dictionary from state
        
        if self.if_mixed:
            if key in self.distrib_dict:
                cumdistr = np.cumsum(self.distrib_dict[key])
                x = np.random.random()
                index = len(cumdistr)-1
                for i in range(len(cumdistr)):
                    if x < cumdistr[i]:
                        index = i
                        break
                return index, self.Q_dict[key][index]
            else:
                return None, 0
        else:
            if key in self.Q_dict:
                if player == 1:
                    action_no, value = np.argmax(self.Q_dict[key]), np.max(self.Q_dict[key])
                else:
                    action_no, value = np.argmin(self.Q_dict[key]), np.min(self.Q_dict[key])
        
                return action_no, value, self.Q_dict[key]
            else:
                return None, 0, []
            
    def set_experience(self,state,action,reward,state_next):
        pass
        
    def to_file(self,filename):
        with open(filename, 'wb') as fp: pickle.dump(self.Q_dict, fp)
        print("strategy with "+str(len(self.Q_dict))+" states is stored in a file "+filename)

    def from_file(self,filename):
        with open(filename, 'rb') as fp: self.Q_dict = pickle.load(fp)
        print("strategy with "+str(len(self.Q_dict))+" states is loaded from a file "+filename)
        
    # methods specific to the Q-learning dictionary method:

    def get_action_value(self,state,action_no):
        key = self.game_obj.state_key(state)    # making key for dictionary from state
        if key in self.Q_dict:
            Qtable = self.Q_dict[key]
            return Qtable[action_no]
        else:
            return 0
        
    def get_Qtable(self,state,player):
        key = self.game_obj.state_key(state)    # making key for dictionary from state
        if key in self.Q_dict:
            Qtable = self.Q_dict[key]
            return Qtable
        else:
            actions = self.game_obj.actions(state,player)
            return np.zeros(len(actions),dtype=float)
        
    def set_action_value(self,state,player,action_no,value):
        key = self.game_obj.state_key(state)    # making key for dictionary from state
        if (key in self.Q_dict) == False:
            actions = self.game_obj.actions(state,player)
            self.Q_dict[key] = np.zeros(len(actions), dtype=float)
        self.Q_dict[key][action_no] = value
            
    def make_epsilon_greedy(self, player, epsilon):
        for key,Qtab in self.Q_dict.items():
            distrib = np.zeros(len(Qtab),dtype=float)
            if player == 1:
                distrib[np.argmax(Qtab)] = 1.0
            else:
                distrib[np.argmin(Qtab)] = 1.0
            self.distrib_dict[key] = distrib
        self.if_mixed = True
        for key,distrib in self.distrib_dict.items():
            number_of_actions = len(distrib)
            distrib_new = np.zeros([len(distrib)], dtype = float)
            if (number_of_actions > 1):
                for i in range(number_of_actions):
                    distrib_new[i] = distrib[i]*(1-epsilon) + epsilon/(number_of_actions-1)
            self.distrib_dict[key] = distrib_new

    def make_pure(self):
        self.if_mixed = False

class Strategy_PolicyNeuro:
    def __init__(self,game_obj):
        self.game_obj = game_obj          # game object - chess
        # ......................

class Strategy_VNeuro:
    def __init__(self,game_obj):
        self.game_obj = game_obj          # game object - chess
        # ......................

# train function using Q table - more general approach: for each agent his opponent is 
# treated as an envoronment. The Q-value update is delayed up to opponent's move due to 
# the potentially random opponent strategy.
# inputs:
#    game_object - Chess
#    players_to_train - name of player for strategy training: 
#        [1] for white, [2] for black or [1,2] for both players
#    strategy_w - strategy for player white (1) - probability of each possible move in each state
#    strategy_b - strategy for player black (2)
#    choose_random - numbers of moves with random action choice:
def board_game_train_Q(game_object, players_to_train, strategy_w=None, strategy_b=None, number_of_games = 1000, choose_random = []):
    
    epsylon = 0.3              # exploration factor
    T = 3
    Tmin = 0.3
    if_softmax = False          # softmax (True) with T or epsilon-greedy (False) with epsilon random factor
    alpha = 0.5                # learning speed factor
    alpha_min = 0.01
    gamma = 0.9                # discount factor
    lambda_ = 0               # fresh factor
    if_sarsa = 0             # 1 - SARSA, 0 - Q-learning
    log_to_file = False

    t1 = time.time()

    dT  = (T - Tmin)/number_of_games                # change of temperature      
    walpha = 1

    if strategy_w == None:
        strategy_w = Strategy_Qdict(game_object)
    if strategy_b == None:
        strategy_b = Strategy_Qdict(game_object)

    if log_to_file:
        f = open("board_game_train_Q.txt","w")
        f.write("... start training by "+str(number_of_games) + " games\n")
        f.write("epsilon = " + str(epsylon) + " softmax = " + str(if_softmax) + " alpha = " + str(alpha) + " gamma = " + str(gamma) + " sarsa = " + str(if_sarsa) + "\n")

    for game_nr in range(number_of_games):                # episodes loop
        if (game_nr*100) % number_of_games == 0: print("game = " + str(game_nr))
        if log_to_file:
            f.write("game = " + str(game_nr) + "\n")
            
        State = game_object.initial_state()               # initial state 
        player = 1                                        # first move by white

        if_end_of_loop = False     
        if_end_of_game = False 
        step_number = 0
        Reward_oppo = 0            

        ToUpdate1 = []
        ToUpdate2 = []

        while (if_end_of_loop == False):                           # episode steps loop
            step_number += 1
            Reward = 0

            if player == 1:
                strategy = strategy_w
            else:
                strategy = strategy_b

            if not if_end_of_game:
                actions = game_object.actions(State, player)
                num_of_actions = len(actions) 
            
                if player in players_to_train:
                    Q_state_actions = strategy.get_Qtable(State,player)
                    ind_Q_best, Q_best, _ = strategy.choose_action(State,player)

                if step_number in choose_random:
                    action_nr = np.random.randint(num_of_actions)
                elif player not in players_to_train:    # using fixed strategy
                    action_nr, value, _ = strategy.choose_action(State,player)
                    if action_nr == None:
                        action_nr = np.random.randint(num_of_actions)
                else:                                   # using learned strategy with exploration
                    if if_softmax:
                        distrib = bfun.softmax((1/T)*Q_state_actions)
                        action_nr = bfun.choose_action(distrib)   
                    elif (np.random.random() < epsylon) | (ind_Q_best == None):    
                        action_nr = np.random.randint(num_of_actions)  
                    else: 
                        action_nr = ind_Q_best                         
                    
                NextState, Reward =  game_object.next_state_and_reward(player,State, actions[action_nr])
                
                if log_to_file:
                    game_object.move_verification(State,actions,NextState,player,f)
                    f.close()
                    f = open("board_game_train_Q.txt","a")
                    f.write(" gracz "+str(player)+" wykonal akcje "+str(action_nr)+" R = "+str(Reward)+"\n")

            if player in players_to_train:
                if step_number > 2:
                    if player == 1:
                        State_prev, action_nr_prev, Reward_prev = ToUpdate1
                        ToUpdate1 = []
                    else:
                        State_prev, action_nr_prev, Reward_prev = ToUpdate2
                        ToUpdate2 = []

                    if if_end_of_game:
                        Q_next = 0
                    else:
                        if if_sarsa:
                            Q_next = Q_state_actions[action_nr]
                        else:
                            Q_next = Q_best

                    Q_prev = strategy.get_action_value(State_prev, action_nr_prev)
                    value = Q_prev + alpha*(Reward_prev + Reward_oppo + gamma*Q_next- Q_prev)
                    strategy.set_action_value(State_prev, player, action_nr_prev, value)
                    
                    if log_to_file:
                        f.write("update Q akcji " + str(action_nr_prev) + " w stanie:\n" + str(State_prev) +\
                                 " R = " + str(Reward_oppo + Reward_prev)+ " Q = "+str(value) + "\n")
                        f.write("Reward_prev = " + str(Reward_prev)+ " Reward_oppo = " + str(Reward_oppo) +\
                                 " Q_next = " + str(Q_next) + "\n")

                if not if_end_of_game:
                    if player == 1:
                        ToUpdate1 = [State, action_nr, Reward]  
                    else:
                        ToUpdate2 = [State, action_nr, Reward]

            State = NextState                                      
            player = 3 - player                             
            Reward_oppo = Reward

            if if_end_of_game:                              
                if_end_of_loop = True                       
                
            if game_object.end_of_game(Reward,step_number,State,action_nr):      
                if_end_of_game = True
                if log_to_file:
                    f.write("chyba koniec gry\n")

        # after the end of game:
        if  len(ToUpdate1) > 0:
            State_prev, action_nr_prev, Reward_prev = ToUpdate1
            Q_prev = strategy_w.get_action_value(State_prev, action_nr_prev)
            value = Q_prev + alpha*(Reward_prev + Reward_oppo - Q_prev)
            strategy_w.set_action_value(State_prev, 1, action_nr_prev, value)
            if log_to_file:
                f.write("update Q akcji " + str(action_nr_prev) + " w stanie:\n" + str(State_prev) + " R = " +\
                         str(Reward_oppo + Reward_prev)+ " Q = "+str(value) + "\n")
                         
        if  len(ToUpdate2) > 0:
            State_prev, action_nr_prev, Reward_prev = ToUpdate2
            Q_prev = strategy_b.get_action_value(State_prev, action_nr_prev)
            value = Q_prev + alpha*(Reward_prev + Reward_oppo - Q_prev)
            strategy_b.set_action_value(State_prev, 2, action_nr_prev, value)
            if log_to_file:
                f.write("update Q akcji " + str(action_nr_prev) + " w stanie:\n" + str(State_prev) + " R = " +\
                         str(Reward_oppo + Reward_prev)+ " Q = "+str(value) + "\n")
                         
        if log_to_file:
            f.write("zamkniecie epizodu " + str(game_nr) + "\n")
            
        T -= dT
        alpha *= walpha

    dt = time.time() - t1
    print("training finished after %.3f sec. (%.3f sec./1000 games)" % (dt, dt*1000/number_of_games) )
    if log_to_file:
        f.close()

    return strategy_w, strategy_b

def board_game_train_Q2(game_object, players_to_train, strategy_w=None, strategy_b=None, number_of_games = 2000):
    
    epsylon = 0.3              
    T = 3
    Tmin = 0.3
    if_softmax = False          
    alpha = 0.5                
    alpha_min = 0.01
    gamma = 0.9                
    lambda_ = 0               
    if_sarsa = 0             
    
    t1 = time.time()
    choose_random = []

    dT  = (T - Tmin)/number_of_games                     
    walpha = np.power(alpha_min/alpha,1/number_of_games)

    if strategy_w == None:
        strategy_w = Strategy_Qdict(game_object)
    if strategy_b == None:
        strategy_b = Strategy_Qdict(game_object)

    print("... start training by "+str(number_of_games) + " games")

    for game_nr in range(number_of_games):                
        if (game_nr*100) % number_of_games == 0: print("game = " + str(game_nr))
        State = game_object.initial_state()               
        player = 1                                        

        if_end_of_loop = False    
        if_end_of_game = False 
        step_number = 0

        while (if_end_of_loop == False):                           
            step_number += 1
            Reward = 0

            if not if_end_of_game:
                actions = game_object.actions(State, player)
                num_of_actions = len(actions)                 

                if player == 1:
                    strategy = strategy_w
                else:  
                    strategy = strategy_b

                if player in players_to_train:       
                    Q_state_actions = strategy.get_Qtable(State,player)
                    ind_Q_best, Q_best, _ = strategy.choose_action(State,player)

                if step_number in choose_random:
                    action_nr = np.random.randint(num_of_actions)
                elif player not in players_to_train:    
                    action_nr, value, _ = strategy.choose_action(State,player)
                    if action_nr == None:
                        action_nr = np.random.randint(num_of_actions)
                else:                                   
                    if if_softmax:
                        distrib = bfun.softmax((1/T)*Q_state_actions)
                        action_nr = bfun.choose_action(distrib)   
                    elif (np.random.random() < epsylon) | (ind_Q_best == None):         
                        action_nr = np.random.randint(num_of_actions)  
                    else: 
                        action_nr = ind_Q_best                         
                            
                NextState, Reward =  game_object.next_state_and_reward(player,State, actions[action_nr])

            if player == 1:
                ToUpdate1 = [State, action_nr, Reward]
            else:
                ToUpdate2 = [State, action_nr, Reward]

            if (3 - player) in players_to_train:
                if step_number > 1:
                    if player == 1:
                        State_prev, action_nr_prev, Reward_prev = ToUpdate2
                        strategy_to_update = strategy_b
                    else:
                        State_prev, action_nr_prev, Reward_prev = ToUpdate1
                        strategy_to_update = strategy_w

                    if if_end_of_game:
                        Q_next = 0
                    else:
                        if if_sarsa:
                            Q_next = Q_state_actions[action_nr]
                        else:
                            Q_next = Q_best

                    Q_prev = strategy_to_update.get_action_value(State_prev, action_nr_prev)
                    value = Q_prev + alpha*(Reward_prev + gamma*Q_next- Q_prev)
                    strategy_to_update.set_action_value(State_prev, 3-player, action_nr_prev, value)
            
            State = NextState                                      
            player = 3 - player                             

            if if_end_of_game:                              
                if_end_of_loop = True                       

            if game_object.end_of_game(Reward,step_number,State,action_nr):      
                if_end_of_game = True
                 
        T -= dT
        alpha *= walpha

    dt = time.time() - t1
    print("training finished after %.3f sec. (%.3f sec./1000 games)" % (dt, dt*1000/number_of_games) )

    return strategy_w, strategy_b


def board_game_test(game_object, strategy_w, strategy_b, number_of_games = 100, choose_random = []):

    num_win_w = 0
    num_win_b = 0
    num_draws = 0
    
    Games = []
    Rewards = []

    for game in range(number_of_games):                   
        State = game_object.initial_state()               
        player = 1                                        
        if_end = False 
        step_number = 0
        States = []
        Actions = []

        States.append(State)

        while (if_end == False):                           
            step_number += 1

            actions = game_object.actions(State, player)   

            if player == 1:
                strategy = strategy_w
            else:
                strategy = strategy_b
            
            action_nr , value = strategy.choose_action(State,player)
            if (action_nr == None) | (step_number in choose_random):
                action_nr = np.random.randint(len(actions))
      
            NextState, Reward =  game_object.next_state_and_reward(player, State, actions[action_nr])

            State = NextState                                        
            Actions.append(actions[action_nr])
            States.append(State)                                     

            player = 3 - player                                      

            if game_object.end_of_game(Reward, step_number,State,action_nr):      
                if_end = True
                if Reward == 1:
                    num_win_w += 1
                elif Reward == -1:
                    num_win_b += 1
                elif Reward == 0:
                    num_draws += 1
                Rewards.append(Reward)
        Games.append([States, Actions])
    return num_win_w, num_win_b, num_draws, Games, Rewards



def experiment_par_train():
    print("\nUCZENIE DWOCH STRATEGII JEDNOCZESNIE\n")
    
    # Wybór planszy szachowej:
    #game = bfun.Chess("szachy_plansza_3x3")
    #game = bfun.Chess("szachy_plansza_4x4")
    #game = bfun.Chess("chess_env/boards/szachy_plansza_5x5")
    #game = bfun.Chess("szachy_plansza_5x3_bez_kroli")
    #game = bfun.Chess("szachy_plansza_5x10")
    game = bfun.Chess("chess_env/boards/szachy_plansza_standardowa")
    #game = bfun.Chess("szachy_plansza_14x14")

    load_q_from_file = False
    save_q_to_file = True


    board_name = os.path.basename(game.initial_board_name)

    strategy_w = Strategy_Qdict(game)
    strategy_b = Strategy_Qdict(game)
    
    if load_q_from_file:
        try:            
            strategy_w.from_file('Q_w_'+board_name+'.pkl')
        except:
            pass
        try:
            strategy_b.from_file('Q_b_'+board_name+'.pkl')
        except:
            pass

    strategy_w, strategy_b = board_game_train_Q(game,players_to_train = [1], 
                                                 strategy_w = strategy_w, strategy_b = strategy_b, 
                                                 number_of_games = 10)

    if save_q_to_file:
        strategy_w.to_file('Q_w_'+board_name+'.pkl')
        strategy_b.to_file('Q_b_'+board_name+'.pkl')

    #bgra.play_with_strategy(game_object = game, strategy = strategy_b, str_player=2)
    bgra.play_with_strategy(game_object = game, strategy = strategy_w, str_player=1)

    print("test stategii uczonych jednocześnie:")
    num_win_w, num_win_b, num_draws, Games, Rewards = board_game_test(game,strategy_w,strategy_b,choose_random=[])
    print("liczby wygranych: white = "+str(num_win_w)+", black = "+str(num_win_b) + ", l.remisów = "+str(num_draws))
    game.print_test_to_file("gry_wyuczonych_strategii.txt",num_win_w, num_win_b, num_draws, Games, Rewards)
    
    print("test stategii white na częściowo losowej black:")
    t = []
    nwin_w = []
    nwin_b = []
    ndraws = []
    for i in range(10):
        epsilon = i/10
        t.append(epsilon)     

        strategy_b.make_epsilon_greedy(player=2,epsilon=epsilon)
        num_win_w, num_win_b, num_draws, Games, Rewards =\
              board_game_test(game, strategy_w, strategy_b)
        game.print_test_to_file("gry_w_vs_losowe_b_epsilon"+str(epsilon)+".txt",\
                                num_win_w, num_win_b, num_draws, Games, Rewards)
        
        nwin_w.append(num_win_w)
        nwin_b.append(num_win_b)
        ndraws.append(num_draws)
    strategy_b.make_pure()
    plt.plot(t,nwin_w,"w",t,nwin_b,"b",t, ndraws,"-")
    plt.title("Chess test results with Nash white strategy and partially random black strategy")
    plt.xlabel("randomness of black strategy (1 - full random)")
    plt.ylabel("number of games")
    plt.legend(["num.of white wins","num.of black wins","num of draws"])
    plt.savefig("test_Nash_w_strategy_random_b_strategy.png")
    fig1 = plt
    plt.show()

    print("test stategii black na częściowo losowej white:")
    t = []
    nwin_w = []
    nwin_b = []
    ndraws = []
    for i in range(10):
        epsilon = i/10
        t.append(epsilon)       
        strategy_w.make_epsilon_greedy(player=1,epsilon=epsilon)
        num_win_w, num_win_b, num_draws, Games, Rewards = \
            board_game_test(game, strategy_w,strategy_b)
        game.print_test_to_file("gry_losowe_w_epsilon"+str(epsilon)+"_vs_b.txt",\
                                num_win_w, num_win_b, num_draws, Games, Rewards)
        nwin_w.append(num_win_w)
        nwin_b.append(num_win_b)
        ndraws.append(num_draws)
    strategy_w.make_pure()
    plt.plot(t,nwin_w,"w",t,nwin_b,"b",t, ndraws,"-")
    plt.title("Chess test results with Nash black strategy and partially random white strategy")
    plt.xlabel("randomness of white strategy (1 - full random)")
    plt.ylabel("number of games")
    plt.legend(["num.of white wins","num.of black wins","num of draws"])
    plt.savefig("test_Nash_b_strategy_random_w_strategy.png")
    plt.show()
    fig2 = plt

    # play with white strategy:
    #bgra.play_with_strategy(game_object = game, strategy = strategy_w, str_player=1)

    # play with black strategy:
    bgra.play_with_strategy(game_object = game, strategy = strategy_b, str_player=2)


# experiment_par_train()  # Commented out - should be called explicitly, not on import