import pygame as pg
import numpy as np
import sys
import time
from pygame.locals import *

class Interface_TicTacToe():
    def __init__(self, game_object):
        self.game_object = game_object
        self.width = 600
        self.height = 600
        self.status_margin = 100
        
        self.d_col = self.width // self.game_object.num_of_columns
        self.d_row = self.height // self.game_object.num_of_rows
        
        pg.init()
        self.screen = pg.display.set_mode((self.width, self.height + self.status_margin))
        pg.display.set_caption("AlphaZero Test: Tic-Tac-Toe")
        
        self.font = pg.font.Font(None, 60)
        self.small_font = pg.font.Font(None, 30)
        self.clock = pg.time.Clock()

    def draw_board(self, State):
        # Fill background with white
        self.screen.fill((255, 255, 255))
        
        line_color = (0, 0, 0)
        line_thickness = 5
        
        # Draw grid lines
        for i in range(1, 3):
            # Horizontal lines
            pg.draw.line(self.screen, line_color, (0, i * self.d_row), (self.width, i * self.d_row), line_thickness)
            # Vertical lines
            pg.draw.line(self.screen, line_color, (i * self.d_col, 0), (i * self.d_col, self.height), line_thickness)

        # Draw X and O pieces
        for r in range(self.game_object.num_of_rows):
            for c in range(self.game_object.num_of_columns):
                val = State.Board[r, c]
                center = (c * self.d_col + self.d_col // 2, r * self.d_row + self.d_row // 2)
                
                if val == 1:
                    # Draw X (Player 1)
                    offset = int(self.d_col * 0.3)
                    pg.draw.line(self.screen, (200, 50, 50),
                                 (center[0] - offset, center[1] - offset),
                                 (center[0] + offset, center[1] + offset), 8)

                    pg.draw.line(self.screen, (200, 50, 50),
                                 (center[0] + offset, center[1] - offset),
                                 (center[0] - offset, center[1] + offset), 8)
                elif val == 2:
                    # Draw O (Player 2)
                    radius = int(self.d_col * 0.3)
                    pg.draw.circle(self.screen, (50, 50, 200), center, radius, 8)

    def draw_status(self, message):
        # Draw dark gray status bar at the bottom
        pg.draw.rect(self.screen, (50, 50, 50), (0, self.height, self.width, self.status_margin))
        text = self.small_font.render(message, True, (255, 255, 255))
        text_rect = text.get_rect(center=(self.width // 2, self.height + self.status_margin // 2))
        self.screen.blit(text, text_rect)
        pg.display.update()

    def user_click(self):
        # Get coordinates of mouse click
        x, y = pg.mouse.get_pos()
        
        # Ignore clicks in the status bar area
        if y > self.height:
            return None, None
            
        col = int(x / self.d_col)
        row = int(y / self.d_row)
        return row, col

    def play_with_strategy(self, strategy, str_player):
        """
        Main loop for Human vs AI interaction.
        str_player: the player number (1 or 2) controlled by the strategy (AI).
        """
        human_player = 3 - str_player    
        
        while True:   # Loop for multiple games
            player = 1                                        
            State = self.game_object.initial_state()               
            end_of_game = False
            step_number = 0
            
            self.draw_board(State)
            self.draw_status("New Game! Player 1 (X) turn.")
            
            while not end_of_game:
                step_number += 1
                actions = self.game_object.actions(State, player)
                action_chosen = None

                if player == human_player:
                    # Human's turn: Wait for a valid mouse click
                    self.draw_status("Your turn! Click an empty square.")
                    legal_action_was_chosen = False
                    
                    while not legal_action_was_chosen:
                        for event in pg.event.get():
                            if event.type == QUIT:
                                pg.quit()
                                sys.exit()
                            elif event.type == MOUSEBUTTONDOWN:
                                row, col = self.user_click()
                                if row is not None and col is not None:
                                    if [row, col] in actions:
                                        action_chosen = [row, col]
                                        legal_action_was_chosen = True
                else:
                    # AI's turn: Get move from strategy
                    self.draw_status("AI is thinking...")
                    time.sleep(0.5) # Small delay for better UX
                    
                    # Assuming strategy.choose_action returns (action_idx, value, Q_tab) or just (action_idx, value)
                    result = strategy.choose_action(State, player)

                    # Handle tuple packing difference (some return 2, some return 3 items)
                    action_idx = result[0] if isinstance(result, (tuple, list)) else result 
                    
                    if action_idx is None:
                        # Fallback to random if AI fails
                        action_idx = np.random.randint(len(actions))
                        
                    action_chosen = actions[action_idx]

                # Apply the chosen action
                NextState, Reward = self.game_object.next_state_and_reward(player, State, action_chosen)
                self.draw_board(NextState)
                
                # Check for win or draw
                if self.game_object.end_of_game(Reward, step_number, NextState):
                    end_of_game = True
                    if Reward == 1:
                        if player == human_player:
                            msg = "You win! Click anywhere to restart."
                        else:
                            msg = "AI wins! Click anywhere to restart."
                    else:
                        msg = "It's a Draw! Click anywhere to restart."
                        
                    self.draw_status(msg)
                    
                player = 3 - player
                State = NextState
                self.clock.tick(30)
                
            # Wait for user to click before starting a new game
            waiting = True
            while waiting:
                for event in pg.event.get():
                    if event.type == QUIT:
                        pg.quit()
                        sys.exit()
                    elif event.type == MOUSEBUTTONDOWN:
                        waiting = False