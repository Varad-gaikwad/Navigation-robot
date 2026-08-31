
import numpy as np
import tensorflow as tf
import pygame
import time

from environment import Env


GRID_SIZE = 10
CELL_SIZE = 60

PANEL_WIDTH = 300
WINDOW_WIDTH = GRID_SIZE * CELL_SIZE + PANEL_WIDTH
WINDOW_HEIGHT = GRID_SIZE * CELL_SIZE

MAX_STEPS = 100
FPS = 5


WHITE = (245, 247, 250)
BLACK = (25, 25, 25)
GRID_COLOR = (210, 214, 220)
BORDER_COLOR = (150, 155, 165)

ROBOT_COLOR = (60, 120, 240)
GOAL_COLOR = (60, 190, 100)
FUTURE_GOAL_COLOR = (180, 220, 190)
COMPLETED_GOAL_COLOR = (170, 175, 180)

OBSTACLE_COLOR = (55, 60, 70)
START_COLOR = (80, 120, 220)

PANEL_COLOR = (235, 238, 243)
CARD_COLOR = (250, 251, 253)

SUCCESS_COLOR = (50, 170, 90)
FAIL_COLOR = (210, 70, 70)


print("Loading trained DQN...")
q_network = tf.keras.models.load_model("best_q_network.keras")
print("Model loaded.")


pygame.init()

screen = pygame.display.set_mode(
    (WINDOW_WIDTH, WINDOW_HEIGHT)
)

pygame.display.set_caption("DQN Robot Navigation")

clock = pygame.time.Clock()

font = pygame.font.Font(None, 27)
small_font = pygame.font.Font(None, 22)
title_font = pygame.font.Font(None, 34)
large_font = pygame.font.Font(None, 42)


def draw_text(text, x, y, font_obj=font, color=BLACK):

    surface = font_obj.render(
        str(text),
        True,
        color
    )

    screen.blit(
        surface,
        (x, y)
    )


def draw_grid():

    for row in range(GRID_SIZE):

        for col in range(GRID_SIZE):

            x = col * CELL_SIZE
            y = row * CELL_SIZE

            rect = pygame.Rect(
                x,
                y,
                CELL_SIZE,
                CELL_SIZE
            )

            pygame.draw.rect(
                screen,
                WHITE,
                rect
            )

            pygame.draw.rect(
                screen,
                GRID_COLOR,
                rect,
                1
            )


def draw_obstacles():

    for row in range(GRID_SIZE):

        for col in range(GRID_SIZE):

            if env.grid[row][col] == 1:

                x = col * CELL_SIZE
                y = row * CELL_SIZE

                rect = pygame.Rect(
                    x + 4,
                    y + 4,
                    CELL_SIZE - 8,
                    CELL_SIZE - 8
                )

                pygame.draw.rect(
                    screen,
                    OBSTACLE_COLOR,
                    rect
                )

                pygame.draw.rect(
                    screen,
                    BORDER_COLOR,
                    rect,
                    1
                )


def draw_start():

    x = env.start_pos[1] * CELL_SIZE
    y = env.start_pos[0] * CELL_SIZE

    pygame.draw.rect(
        screen,
        START_COLOR,
        (
            x + 8,
            y + 8,
            CELL_SIZE - 16,
            CELL_SIZE - 16
        ),
        3
    )

    draw_text(
        "S",
        x + 23,
        y + 18,
        font,
        START_COLOR
    )


def draw_goals():

    for i, goal in enumerate(env.goals):

        x = goal[1] * CELL_SIZE
        y = goal[0] * CELL_SIZE

        if i < env.current_goal_index:

            color = COMPLETED_GOAL_COLOR

        elif i == env.current_goal_index:

            color = GOAL_COLOR

        else:

            color = FUTURE_GOAL_COLOR

        pygame.draw.rect(
            screen,
            color,
            (
                x + 7,
                y + 7,
                CELL_SIZE - 14,
                CELL_SIZE - 14
            )
        )

        pygame.draw.rect(
            screen,
            BORDER_COLOR,
            (
                x + 7,
                y + 7,
                CELL_SIZE - 14,
                CELL_SIZE - 14
            ),
            1
        )

        draw_text(
            str(i + 1),
            x + 24,
            y + 18,
            font,
            BLACK
        )


def draw_robot():

    x = env.robot_pos[1] * CELL_SIZE
    y = env.robot_pos[0] * CELL_SIZE

    center = (
        x + CELL_SIZE // 2,
        y + CELL_SIZE // 2
    )

    pygame.draw.circle(
        screen,
        ROBOT_COLOR,
        center,
        CELL_SIZE // 3
    )

    pygame.draw.circle(
        screen,
        WHITE,
        center,
        CELL_SIZE // 3,
        2
    )


def draw_panel(
    steps,
    total_reward,
    action_name,
    q_values,
    status
):

    panel_x = GRID_SIZE * CELL_SIZE

    pygame.draw.rect(
        screen,
        PANEL_COLOR,
        (
            panel_x,
            0,
            PANEL_WIDTH,
            WINDOW_HEIGHT
        )
    )

    pygame.draw.line(
        screen,
        BORDER_COLOR,
        (
            panel_x,
            0
        ),
        (
            panel_x,
            WINDOW_HEIGHT
        ),
        2
    )

    x = panel_x + 20

    draw_text(
        "DQN NAVIGATION",
        x,
        20,
        title_font
    )

    pygame.draw.line(
        screen,
        BORDER_COLOR,
        (
            x,
            58
        ),
        (
            panel_x + PANEL_WIDTH - 20,
            58
        ),
        1
    )

    draw_text(
        "SIMULATION",
        x,
        75,
        small_font,
        (90, 95, 105)
    )

    if status == "SUCCESS":

        status_color = SUCCESS_COLOR

    elif status == "FAILED":

        status_color = FAIL_COLOR

    else:

        status_color = ROBOT_COLOR

    draw_text(
        status,
        x,
        98,
        font,
        status_color
    )

    y = 140

    draw_text(
        "PROGRESS",
        x,
        y,
        small_font,
        (90, 95, 105)
    )

    y += 30

    completed = min(
        env.current_goal_index,
        len(env.goals)
    )

    draw_text(
        f"Goals: {completed}/{len(env.goals)}",
        x,
        y
    )

    y += 28

    current_goal = min(
        env.current_goal_index + 1,
        len(env.goals)
    )

    draw_text(
        f"Current goal: {current_goal}",
        x,
        y
    )

    y += 28

    draw_text(
        f"Steps: {steps}/{MAX_STEPS}",
        x,
        y
    )

    y += 28

    draw_text(
        f"Reward: {total_reward:.1f}",
        x,
        y
    )

    y += 45

    draw_text(
        "POSITION",
        x,
        y,
        small_font,
        (90, 95, 105)
    )

    y += 30

    draw_text(
        f"Robot: {env.robot_pos}",
        x,
        y
    )

    y += 28

    draw_text(
        f"Target: {env.goal_pos}",
        x,
        y
    )

    distance = (
        abs(
            env.robot_pos[0] -
            env.goal_pos[0]
        )
        +
        abs(
            env.robot_pos[1] -
            env.goal_pos[1]
        )
    )

    y += 28

    draw_text(
        f"Distance: {distance}",
        x,
        y
    )

    y += 45

    draw_text(
        "DQN DECISION",
        x,
        y,
        small_font,
        (90, 95, 105)
    )

    y += 30

    draw_text(
        f"Action: {action_name}",
        x,
        y
    )

    y += 30

    actions = [
        "UP",
        "DOWN",
        "RIGHT",
        "LEFT"
    ]

    for i, action in enumerate(actions):

        value = float(q_values[i])

        draw_text(
            f"{action:<6} {value:8.2f}",
            x,
            y,
            small_font
        )

        y += 24

    y += 15

    max_q = float(
        np.max(q_values)
    )

    draw_text(
        f"Max Q-value: {max_q:.2f}",
        x,
        y,
        small_font
    )

    y += 40

    draw_text(
        "R  Restart",
        x,
        y,
        small_font
    )

    y += 24

    draw_text(
        "ESC  New map",
        x,
        y,
        small_font
    )

    y += 24

    draw_text(
        "ENTER  Continue",
        x,
        y,
        small_font
    )


def draw_environment(
    steps=0,
    total_reward=0,
    action_name="None",
    q_values=np.zeros(4),
    status="RUNNING"
):

    screen.fill(WHITE)

    draw_grid()
    draw_obstacles()
    draw_goals()
    draw_start()
    draw_robot()

    draw_panel(
        steps,
        total_reward,
        action_name,
        q_values,
        status
    )

    pygame.display.flip()


def select_map():

    start_pos = None
    goals = []
    obstacles = []

    selecting = True

    print()
    print("================================")
    print("NEW MAP")
    print("================================")
    print("Click a cell = START")
    print("Click 3 cells = GOALS")
    print("Click cells = OBSTACLES")
    print("Click obstacle again = REMOVE")
    print("BACKSPACE = remove last obstacle")
    print("ENTER = start simulation")
    print("ESC = quit")
    print()

    while selecting:

        for event in pygame.event.get():

            if event.type == pygame.QUIT:

                pygame.quit()
                raise SystemExit

            if event.type == pygame.MOUSEBUTTONDOWN:

                mouse_x, mouse_y = event.pos

                if mouse_x >= GRID_SIZE * CELL_SIZE:
                    continue

                col = mouse_x // CELL_SIZE
                row = mouse_y // CELL_SIZE

                selected_pos = [
                    row,
                    col
                ]

                if start_pos is None:

                    start_pos = selected_pos

                    print(
                        "Start:",
                        start_pos
                    )

                elif len(goals) < 3:

                    if selected_pos == start_pos:
                        continue

                    if selected_pos in goals:
                        continue

                    goals.append(
                        selected_pos
                    )

                    print(
                        "Goal",
                        len(goals),
                        ":",
                        selected_pos
                    )

                else:

                    if selected_pos == start_pos:
                        continue

                    if selected_pos in goals:
                        continue

                    if selected_pos in obstacles:

                        obstacles.remove(
                            selected_pos
                        )

                        print(
                            "Removed obstacle:",
                            selected_pos
                        )

                    else:

                        obstacles.append(
                            selected_pos
                        )

                        print(
                            "Obstacle:",
                            selected_pos,
                            "| Total:",
                            len(obstacles)
                        )

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_BACKSPACE:

                    if obstacles:

                        removed = obstacles.pop()

                        print(
                            "Removed:",
                            removed
                        )

                if event.key == pygame.K_RETURN:

                    if start_pos is None:

                        print(
                            "Select START first."
                        )

                    elif len(goals) < 3:

                        print(
                            "Select",
                            3 - len(goals),
                            "more goal(s)."
                        )

                    else:

                        selecting = False

                if event.key == pygame.K_ESCAPE:

                    pygame.quit()
                    raise SystemExit

        screen.fill(WHITE)

        draw_grid()

        if start_pos is not None:

            x = start_pos[1] * CELL_SIZE
            y = start_pos[0] * CELL_SIZE

            pygame.draw.rect(
                screen,
                START_COLOR,
                (
                    x + 8,
                    y + 8,
                    CELL_SIZE - 16,
                    CELL_SIZE - 16
                ),
                3
            )

            draw_text(
                "S",
                x + 23,
                y + 18,
                font,
                START_COLOR
            )

        for i, goal in enumerate(goals):

            x = goal[1] * CELL_SIZE
            y = goal[0] * CELL_SIZE

            pygame.draw.rect(
                screen,
                GOAL_COLOR,
                (
                    x + 7,
                    y + 7,
                    CELL_SIZE - 14,
                    CELL_SIZE - 14
                )
            )

            draw_text(
                str(i + 1),
                x + 24,
                y + 18,
                font,
                BLACK
            )

        for obstacle in obstacles:

            row = obstacle[0]
            col = obstacle[1]

            x = col * CELL_SIZE
            y = row * CELL_SIZE

            pygame.draw.rect(
                screen,
                OBSTACLE_COLOR,
                (
                    x + 4,
                    y + 4,
                    CELL_SIZE - 8,
                    CELL_SIZE - 8
                )
            )

        panel_x = GRID_SIZE * CELL_SIZE

        pygame.draw.rect(
            screen,
            PANEL_COLOR,
            (
                panel_x,
                0,
                PANEL_WIDTH,
                WINDOW_HEIGHT
            )
        )

        draw_text(
            "MAP SETUP",
            panel_x + 20,
            25,
            title_font
        )

        draw_text(
            "Start",
            panel_x + 20,
            90,
            small_font
        )

        draw_text(
            str(start_pos),
            panel_x + 20,
            115
        )

        draw_text(
            "Goals",
            panel_x + 20,
            160,
            small_font
        )

        draw_text(
            f"{len(goals)}/3 selected",
            panel_x + 20,
            185
        )

        for i, goal in enumerate(goals):

            draw_text(
                f"Goal {i + 1}: {goal}",
                panel_x + 20,
                220 + i * 28,
                small_font
            )

        draw_text(
            "Obstacles",
            panel_x + 20,
            315,
            small_font
        )

        draw_text(
            str(len(obstacles)),
            panel_x + 20,
            340
        )

        draw_text(
            "CONTROLS",
            panel_x + 20,
            410,
            small_font,
            (90, 95, 105)
        )

        draw_text(
            "Mouse: select cells",
            panel_x + 20,
            440,
            small_font
        )

        draw_text(
            "Backspace: remove",
            panel_x + 20,
            468,
            small_font
        )

        draw_text(
            "Enter: start",
            panel_x + 20,
            496,
            small_font
        )

        draw_text(
            "ESC: quit",
            panel_x + 20,
            524,
            small_font
        )

        pygame.display.flip()

        clock.tick(60)

    return (
        start_pos,
        goals,
        obstacles
    )


def reset_episode():

    env.robot_pos = env.start_pos.copy()
    env.current_goal_index = 0
    env.goal_pos = env.goals[0].copy()
    env.previous_action = -1


def run_episode():

    state = env.get_state()
    state = np.array(
        state,
        dtype=np.float32
    )

    done = False
    steps = 0
    total_reward = 0

    action_name = "None"
    q_values = np.zeros(4)

    draw_environment(
        steps,
        total_reward,
        action_name,
        q_values
    )

    time.sleep(1)

    while not done and steps < MAX_STEPS:

        for event in pygame.event.get():

            if event.type == pygame.QUIT:

                pygame.quit()
                raise SystemExit

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:
                    return "menu"

                if event.key == pygame.K_r:

                    reset_episode()

                    return "restart"

        state_input = np.expand_dims(
            state,
            axis=0
        )

        q_values = q_network(
            state_input,
            training=False
        ).numpy()[0]

        action = int(
            np.argmax(q_values)
        )

        action_names = {
            0: "UP",
            1: "DOWN",
            2: "RIGHT",
            3: "LEFT"
        }

        action_name = action_names[action]

        next_state, reward, done = env.step(
            action
        )

        next_state = np.array(
            next_state,
            dtype=np.float32
        )

        state = next_state

        total_reward += reward
        steps += 1

        print(
            "Step:",
            steps,
            "| Action:",
            action_name,
            "| Position:",
            env.robot_pos,
            "| Goal:",
            env.current_goal_index + 1
        )

        draw_environment(
            steps,
            total_reward,
            action_name,
            q_values
        )

        clock.tick(FPS)

    if done:

        status = "SUCCESS"

    else:

        status = "FAILED"

    draw_environment(
        steps,
        total_reward,
        action_name,
        q_values,
        status
    )

    print()
    print("================================")
    print(status)
    print("Steps:", steps)
    print("Total reward:", total_reward)
    print(
        "Goals completed:",
        min(
            env.current_goal_index,
            len(env.goals)
        ),
        "/",
        len(env.goals)
    )
    print("================================")

    return "finished"


def simulation_screen():

    while True:

        result = run_episode()

        if result == "menu":

            return

        if result == "restart":

            continue

        while True:

            for event in pygame.event.get():

                if event.type == pygame.QUIT:

                    pygame.quit()
                    raise SystemExit

                if event.type == pygame.KEYDOWN:

                    if event.key == pygame.K_r:

                        reset_episode()

                        break

                    if event.key == pygame.K_ESCAPE:

                        return

                    if event.key == pygame.K_RETURN:

                        return

            else:

                pygame.display.flip()
                clock.tick(30)
                continue

            break


while True:

    start_pos, goals, obstacles = select_map()

    env = Env(
        start_pos,
        goals[0],
        len(obstacles)
    )

    env.set_goals(goals)

    env.grid = np.zeros(
        (
            GRID_SIZE,
            GRID_SIZE
        ),
        dtype=int
    )

    for obstacle in obstacles:

        row = obstacle[0]
        col = obstacle[1]

        env.grid[row][col] = 1

    simulation_screen()


pygame.quit()
