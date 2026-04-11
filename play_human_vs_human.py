import argparse
import sys
import time

import pygame as pg
from pygame.locals import K_BACKSPACE, K_END, K_ESCAPE, K_SPACE, KEYDOWN, MOUSEBUTTONDOWN, QUIT

from chess_env.chess_game import Chess
from chess_env.chess_interface import Interface_Chess


def _draw_move(interface, game, state, action):
    """Draw a single move on board, including castling and en passant cleanup."""
    interface.draw_empty(action[1], action[2])
    interface.draw_empty(action[3], action[4])

    # En passant: pawn moved diagonally to an empty target square in previous state.
    if ((action[0] == game.Pawn) or (action[0] == game.Pawn + game.BlackShift)) and \
       (action[2] != action[4]) and (state.Board[action[3], action[4]] == 0):
        interface.draw_empty(action[1], action[4])

    if len(action) > 5:
        interface.draw_piece(action[3], action[4], action[5])
    else:
        interface.draw_piece(action[3], action[4], action[0])

    # Castling: draw extra rook move.
    if (action[0] == game.King) and (action[2] == 4) and (action[4] == 6):
        interface.draw_empty(action[1], 7)
        interface.draw_piece(action[1], 5, game.Rook)
    if (action[0] == game.King) and (action[2] == 4) and (action[4] == 2):
        interface.draw_empty(action[1], 0)
        interface.draw_piece(action[1], 3, game.Rook)
    if (action[0] == game.King + game.BlackShift) and (action[2] == 4) and (action[4] == 6):
        interface.draw_empty(action[1], 7)
        interface.draw_piece(action[1], 5, game.Rook + game.BlackShift)
    if (action[0] == game.King + game.BlackShift) and (action[2] == 4) and (action[4] == 2):
        interface.draw_empty(action[1], 0)
        interface.draw_piece(action[1], 3, game.Rook + game.BlackShift)


def _choose_human_action(interface, game, state, player, actions):
    """Return selected action index, restart flag, quit flag."""
    actions_point_from = []
    actions_point_fromto = []

    for act in actions:
        if [act[1], act[2]] not in actions_point_from:
            actions_point_from.append([act[1], act[2]])
        actions_point_fromto.append([act[1], act[2], act[3], act[4]])

    for act in actions_point_from:
        interface.draw_frame(act[0], act[1], color=(0, 255, 0))

    choose_promotion = False
    promotion_piece = None
    starting_point = None
    fromto = None
    legal_action_was_choosen = False
    action_nr = None

    while not legal_action_was_choosen:
        row = col = None

        for event in pg.event.get():
            if event.type == QUIT:
                return None, False, True

            if event.type == MOUSEBUTTONDOWN:
                row, col = interface.user_click()

            elif event.type == KEYDOWN:
                if (event.key == K_BACKSPACE) or (event.key == K_END):
                    return None, True, False

                if event.key == K_ESCAPE:
                    choose_promotion = False
                    promotion_piece = None
                    if starting_point is not None:
                        sr, sc = starting_point
                        interface.draw_piece(sr, sc, state.Board[sr, sc])
                    starting_point = None
                    fromto = None

                elif event.key == K_SPACE:
                    row, col = interface.user_click()

                elif (event.key == pg.K_q) and (fromto is not None):
                    promotion_piece = game.Queen + (game.BlackShift if player == 2 else 0)
                    row, col = fromto[2], fromto[3]
                    choose_promotion = False

                elif (event.key == pg.K_k) and (fromto is not None):
                    promotion_piece = game.Knight + (game.BlackShift if player == 2 else 0)
                    row, col = fromto[2], fromto[3]
                    choose_promotion = False

                elif (event.key == pg.K_r) and (fromto is not None):
                    promotion_piece = game.Rook + (game.BlackShift if player == 2 else 0)
                    row, col = fromto[2], fromto[3]
                    choose_promotion = False

                elif (event.key == pg.K_b) and (fromto is not None):
                    promotion_piece = game.Bishop + (game.BlackShift if player == 2 else 0)
                    row, col = fromto[2], fromto[3]
                    choose_promotion = False

        if (row is not None) and (col is not None) and ([row, col] in actions_point_from):
            if starting_point is not None:
                sr, sc = starting_point
                interface.draw_piece(sr, sc, state.Board[sr, sc])
                for act in actions_point_fromto:
                    if [sr, sc] == [act[0], act[1]]:
                        interface.draw_frame(act[2], act[3], color=None)

            starting_point = [row, col]
            interface.draw_piece(row, col, state.Board[row, col], red_contour=True)
            interface.draw_status(player, None, "Wybierz pole docelowe. ESC - anuluj wybor")
            for act in actions_point_fromto:
                if [row, col] == [act[0], act[1]]:
                    interface.draw_frame(act[2], act[3], color=(0, 0, 255))

        elif choose_promotion:
            continue

        elif starting_point is not None:
            if (row is not None) and (col is not None) and \
               ([starting_point[0], starting_point[1], row, col] in actions_point_fromto):
                fromto = [starting_point[0], starting_point[1], row, col]

                candidate_indices = []
                for i, act in enumerate(actions):
                    if fromto == [act[1], act[2], act[3], act[4]]:
                        candidate_indices.append(i)

                if any(len(actions[i]) > 5 for i in candidate_indices):
                    if promotion_piece is None:
                        choose_promotion = True
                        interface.draw_status(
                            player,
                            None,
                            "Promocja: Q-hetman, K-skoczek, R-wieza, B-goniec"
                        )
                    else:
                        for i in candidate_indices:
                            if (len(actions[i]) > 5) and (promotion_piece == actions[i][5]):
                                action_nr = i
                                legal_action_was_choosen = True
                                break
                else:
                    action_nr = candidate_indices[0]
                    legal_action_was_choosen = True

                for act in actions_point_fromto:
                    if [starting_point[0], starting_point[1]] == [act[0], act[1]]:
                        interface.draw_frame(act[2], act[3], color=None)

        pg.display.update()
        interface.CLOCK.tick(30)

    for act in actions_point_from:
        interface.draw_frame(act[0], act[1], color=None)

    return action_nr, False, False


def play_human_vs_human(board_name):
    game = Chess(board_name)
    interface = Interface_Chess(game)

    end_of_interaction = False
    while not end_of_interaction:
        player = 1
        winner = None
        state = game.initial_state()
        end_of_game = False
        step_number = 0

        interface.graphical_board_initiating()
        interface.draw_status(player, None, "Tryb: gracz vs gracz")
        print("Nowa gra! (BACKSPACE lub END - nowa partia, ESC - anuluj wybor)")

        while not end_of_game:
            step_number += 1
            actions = game.actions(state, player)

            if len(actions) == 0:
                winner = 0
                if not state.white_king_mated and not state.black_king_mated:
                    print("PAT! Brak legalnych ruchow.")
                end_of_game = True
                break

            action_nr, restart_game, quit_app = _choose_human_action(interface, game, state, player, actions)

            if quit_app:
                end_of_interaction = True
                break

            if restart_game:
                end_of_game = True
                break

            print("step " + str(step_number) + ", " + game.action_to_string(actions[action_nr]))
            next_state, reward = game.next_state_and_reward(player, state, actions[action_nr])
            _draw_move(interface, game, state, actions[action_nr])

            if game.end_of_game(reward, step_number, state, action_nr):
                end_of_game = True
                if reward == 1:
                    winner = 1
                elif reward == -1:
                    winner = 2
                else:
                    winner = 0
                    if next_state.white_blocked or next_state.black_blocked:
                        status_extra_text = "Tryb: gracz vs gracz | PAT"
                        print("PAT! Gra zakonczona remisem.")
                interface.draw_end_of_game(next_state, player, winner)

            interface.draw_status(3 - player, winner, status_extra_text)
            pg.display.update()

            player = 3 - player
            state = next_state

        if not end_of_interaction:
            time.sleep(0.5)

    pg.quit()
    sys.exit()


def main():
    parser = argparse.ArgumentParser(description="Human vs Human chess on custom board")
    parser.add_argument(
        "--board",
        default="chess_env/boards/szachy_plansza_standardowa",
        help="Board file path without .txt extension"
    )
    args = parser.parse_args()

    play_human_vs_human(args.board)


if __name__ == "__main__":
    main()
