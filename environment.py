import numpy as np
from collections import deque

class Env:
    def __init__(self, start_pos, goal_pos, num_obstacles):

        self.robot_pos= start_pos
        self.goal_pos= goal_pos
        self.goals = [goal_pos.copy()]
        self.current_goal_index = 0

        self.start_pos = start_pos.copy()

        self.num_obstacles=num_obstacles

        self.actions={0:"UP",
                        1:"DOWN",
                        2:"RIGHT",
                        3:"LEFT"}
        
        self.reward=0

        self.previous_action= -1


    def set_goals(self, goals):
        self.goals = [goal.copy() for goal in goals]
        self.current_goal_index=0
        self.goal_pos = self.goals[0].copy()#This means the DQN initially targets goal 1.


    def is_solvable(self):

        current_start = self.start_pos.copy()

        for goal in self.goals:

            queue = deque([tuple(current_start)])
            visited = {tuple(current_start)}

            found = False

            while queue:

                x, y = queue.popleft()

                if [x, y] == goal:
                   found = True
                   break

                moves = [
                (-1, 0),
                (1, 0),
                (0, 1),
                (0, -1)
            ]

                for dx, dy in moves:

                  new_x = x + dx
                  new_y = y + dy

                  if not (0 <= new_x < 10 and 0 <= new_y < 10):
                         continue

                  if self.grid[new_x][new_y] == 1:
                    continue

                  new_pos = (new_x, new_y)

                  if new_pos in visited:
                    continue

                  visited.add(new_pos)
                  queue.append(new_pos)

            if not found:
                return False

        # Next search starts from this goal
            current_start = goal.copy()

        return True

    def generate_map(self):
        while True:

            self.grid= np.zeros((10,10),dtype=int)#We use a 10x10 grid.

            obstacles=0
            while obstacles < self.num_obstacles:
                x=np.random.randint(0,10)
                y=np.random.randint(0,10)

                if [x, y] == self.start_pos:
                     continue

                if [x, y] in self.goals:
                     continue

                if self.grid[x,y] == 1:
                     continue

                self.grid[x,y]=1
                obstacles+=1

            if self.is_solvable(): #If true then it leaves the function and uses the map, else generates a new solvable map.
                return 

    def get_state(self):

        # Robot position
        x1 = self.robot_pos[0] / 9
        y1 = self.robot_pos[1] / 9
        # Goal position
        x2 = self.goal_pos[0] / 9
        y2 = self.goal_pos[1] / 9

        # 9x9 local vision
        local_view = []
        robot_x = self.robot_pos[0]
        robot_y = self.robot_pos[1]

        for dx in [-2,-1, 0, 1,2]:
            for dy in [-2,-1, 0, 1,2]:

                x = robot_x + dx
                y = robot_y + dy
                # Outside grid = obstacle
                if not (0 <= x < 10 and 0 <= y < 10):
                    local_view.append(1)
                # Obstacle
                elif self.grid[x][y] == 1:
                    local_view.append(1)
                # Empty
                else:
                    local_view.append(0)

        previous_action=[0,0,0,0]

        if self.previous_action != -1:
            previous_action[self.previous_action] = 1

        state = [x1,y1,x2,y2] + local_view + previous_action

        return state
    

    
    def step(self,action):
            old_distance = abs(self.robot_pos[0] - self.goal_pos[0]) + abs(self.robot_pos[1] - self.goal_pos[1])
            #Chooses action
            if action == 0:
                self.new_pos= [self.robot_pos[0]-1, self.robot_pos[1]]
            elif action == 1:
                self.new_pos=[self.robot_pos[0] + 1, self.robot_pos[1]]
            elif action == 2:
                self.new_pos= [self.robot_pos[0],self.robot_pos[1] + 1]
            elif action == 3:
                self.new_pos= [self.robot_pos[0], self.robot_pos[1] - 1]
            
            #If the robot goes out of the grid.
            if not( 0<= self.new_pos[0] < 10 and 
                   0<= self.new_pos[1] < 10):  
                self.reward= -10
                done= False

                return self.get_state(), self.reward, done

            #If obstacle is detected pos wont change else it will change.
            if self.grid[self.new_pos[0]][self.new_pos[1]]== 1 :#If it crashes into obstacle
                self.reward= -10
                done=False

                return self.get_state(), self.reward, done
            
            self.robot_pos=self.new_pos 
            
            new_distance= abs(self.robot_pos[0] - self.goal_pos[0])+ abs(self.robot_pos[1]-self.goal_pos[1])
            #If the robot moves towards the goal than -1 reward and if it moves away from the goal -2 reward.

            if self.robot_pos == self.goal_pos: #If it reaches goal
                self.reward= 100
                self.current_goal_index+= 1

                if self.current_goal_index < len(self.goals):
                    self.goal_pos = self.goals[self.current_goal_index].copy()

                    done= False
                    self.previous_action= action

                    return self.get_state(), self.reward, done
                else:

                    done=True
                    self.previous_action = action

                    return self.get_state(), self.reward, done

            self.reward= -1

            if new_distance < old_distance:
                    self.reward += 3

            else:
                    self.reward -= 2

            if (
              (self.previous_action == 0 and action == 1) or
              (self.previous_action == 1 and action == 0) or
              (self.previous_action == 2 and action == 3) or
              (self.previous_action == 3 and action == 2)):
                self.reward -= 5

            self.previous_action = action

            done=False

            return self.get_state(), self.reward, done

    def reset(self):

       self.reward = 0
       self.robot_pos = self.start_pos.copy()
       self.generate_map()

       self.goal_pos= self.goals[0].copy()
       self.current_goal_index=0

       self.previous_action = -1

       return self.get_state()



            

            
                 
       