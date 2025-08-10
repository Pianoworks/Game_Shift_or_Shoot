import pygame
import math
import sys
import os
import random

# --- Pygameの初期化 ---
pygame.init()

# --- 定数の設定 (横向き盤面) ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
BOARD_WIDTH = 700
BOARD_HEIGHT = 350
PLATE_SIZE = 80      # 正方形の板のサイズ
WINNING_SCORE = 5

# --- 色の定義 ---
WHITE = (255, 255, 255)
PLAYER_1_COLOR = (0, 100, 255)  # 青色 (左陣営)
PLAYER_2_COLOR = (255, 50, 50)  # 赤色 (右陣営)

### 修正点1: 物理パラメータを更新 ###
# --- 物理エンジンの設定 ---
FRICTION_X = 0.95
FRICTION_Y = 0.95
ANGULAR_FRICTION = 0.95
SIDE_FRICTION = 0.7
STOP_THRESHOLD = 0.1
ANGULAR_STOP_THRESHOLD = 0.1
POWER_MULTIPLIER = 0.10
TORQUE_MULTIPLIER = 0.0008
COLLISION_ELASTICITY = 0.85
PLATE_INERTIA = (PLATE_SIZE ** 2) / 6
# 追加: 衝突時の回転力を調整する係数
COLLISION_TORQUE_FACTOR = 0.02

class Plate:
    def __init__(self, owner, plate_id, x, y, texture, angle):
        self.owner = owner
        self.id = plate_id
        self.x, self.y = x, y
        self.vx, self.vy = 0, 0
        self.angle = angle
        self.angular_velocity = 0
        self.size = PLATE_SIZE
        self.original_image = pygame.transform.scale(texture, (self.size, self.size))
        self.image = self.original_image.copy()
        self.rect = self.original_image.get_rect(center=(self.x, self.y))
        self.corners = []
        self.update_corners()

    def update_corners(self):
        corners = []
        half_size = self.size / 2
        local_corners = [
            (-half_size, -half_size), (half_size, -half_size),
            (half_size, half_size), (-half_size, half_size)
        ]
        rad = math.radians(self.angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        for x, y in local_corners:
            rotated_x = x * cos_a - y * sin_a + self.x
            rotated_y = x * sin_a + y * cos_a + self.y
            corners.append(pygame.math.Vector2(rotated_x, rotated_y))
        self.corners = corners

    def draw(self, surface):
        image_to_rotate = self.original_image.copy()
        color_overlay = pygame.Surface(image_to_rotate.get_size(), pygame.SRCALPHA)
        color = PLAYER_1_COLOR if self.owner == 1 else PLAYER_2_COLOR
        alpha = int(255 * 0.1)
        color_overlay.fill((*color, alpha))
        image_to_rotate.blit(color_overlay, (0, 0))
        rotated_image = pygame.transform.rotate(image_to_rotate, self.angle)
        self.image = rotated_image
        self.rect = self.image.get_rect(center=(self.x, self.y))
        surface.blit(self.image, self.rect)

    def update(self):
        if self.is_moving():
            self.x += self.vx
            self.y += self.vy
            self.angle += self.angular_velocity
            self.angle %= 360
            self.update_corners()

            on_board = self.rect.right > (SCREEN_WIDTH - BOARD_WIDTH) / 2 and \
                       self.rect.left < (SCREEN_WIDTH + BOARD_WIDTH) / 2 and \
                       self.rect.bottom > (SCREEN_HEIGHT - BOARD_HEIGHT) / 2 and \
                       self.rect.top < (SCREEN_HEIGHT + BOARD_HEIGHT) / 2

            if not on_board:
                self.vx *= SIDE_FRICTION
                self.vy *= SIDE_FRICTION
                self.angular_velocity *= ANGULAR_FRICTION
            else:
                self.vx *= FRICTION_X
                self.vy *= FRICTION_Y
                self.angular_velocity *= ANGULAR_FRICTION

            if abs(self.vx) < STOP_THRESHOLD: self.vx = 0
            if abs(self.vy) < STOP_THRESHOLD: self.vy = 0
            if abs(self.angular_velocity) < ANGULAR_STOP_THRESHOLD: self.angular_velocity = 0

    def is_moving(self):
        return self.vx != 0 or self.vy != 0 or self.angular_velocity != 0

    def apply_force_and_torque(self, force_vector, offset_vector):
        self.vx = force_vector[0] * POWER_MULTIPLIER
        self.vy = force_vector[1] * POWER_MULTIPLIER
        torque = offset_vector[0] * force_vector[1] - offset_vector[1] * force_vector[0]
        self.angular_velocity += torque * TORQUE_MULTIPLIER

class SosGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Shift or Shoot")
        self.font = pygame.font.Font(None, 48)
        self.small_font = pygame.font.Font(None, 32)
        self.clock = pygame.time.Clock()
        self.plates = []
        self.scores = {1: 0, 2: 0}
        self.current_player = 1
        self.selected_plate = None
        self.drag_start_pos = None
        self.click_offset = None
        self.game_state = "PLAYER_TURN"
        self._load_textures()
        self._setup_board()

    def _load_textures(self):
        try:
            wood_texture = pygame.image.load('wood_texture.jpg').convert()
            self.marble_texture = pygame.image.load('marble_texture.jpg').convert()
            self.plate_textures = [pygame.image.load(f'plate{i}.jpg').convert_alpha() for i in range(1, 7)]
        except pygame.error as e:
            print(f"エラー: 画像ファイルの読み込みに失敗しました。")
            print(f"Pygameファイルと同じフォルダに 'wood_texture.jpg', 'marble_texture.jpg', 'plate1.jpg' ... 'plate6.jpg' が全て存在するか確認してください。")
            print(f"Pygameからの詳細エラー: {e}")
            pygame.quit()
            sys.exit()
        self.background_image = pygame.transform.scale(wood_texture, (SCREEN_WIDTH, SCREEN_HEIGHT))
        self.board_image = pygame.transform.scale(self.marble_texture, (BOARD_WIDTH, BOARD_HEIGHT))

    def _setup_board(self):
        self.plates.clear()
        for i in range(1, 4):
            self._place_randomly(1, i, self.plate_textures[i - 1])
        for i in range(1, 4):
            self._place_randomly(2, i, self.plate_textures[i - 1 + 3])

    def _place_randomly(self, player, plate_id, texture):
        board_x_start = (SCREEN_WIDTH - BOARD_WIDTH) / 2
        board_y_start = (SCREEN_HEIGHT - BOARD_HEIGHT) / 2
        half_board_width = BOARD_WIDTH / 2
        x_range_start, x_range_end = (board_x_start, board_x_start + half_board_width) if player == 1 else (board_x_start + half_board_width, board_x_start + BOARD_WIDTH)
        while True:
            x = random.uniform(x_range_start + PLATE_SIZE / 2, x_range_end - PLATE_SIZE / 2)
            y = random.uniform(board_y_start + PLATE_SIZE / 2, board_y_start + BOARD_HEIGHT - PLATE_SIZE / 2)
            angle = random.uniform(0, 90)
            temp_plate = Plate(player, plate_id, x, y, texture, angle)
            if not any(self.check_collision_sat(existing_plate, temp_plate)[0] for existing_plate in self.plates):
                self.plates.append(temp_plate)
                break

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(), sys.exit()
            if self.game_state == "PLAYER_TURN":
                self.handle_mouse_events(event)
            elif self.game_state == "GAME_OVER" and event.type == pygame.MOUSEBUTTONDOWN:
                self.__init__()

    def handle_mouse_events(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for plate in self.plates:
                if plate.owner == self.current_player and plate.rect.collidepoint(event.pos):
                    try:
                        if plate.image.get_at((event.pos[0] - plate.rect.left, event.pos[1] - plate.rect.top))[3] > 0:
                            self.selected_plate = plate
                            self.drag_start_pos = event.pos
                            self.click_offset = (event.pos[0] - plate.x, event.pos[1] - plate.y)
                            break
                    except IndexError: continue
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.selected_plate:
            force_vector = (self.drag_start_pos[0] - event.pos[0], self.drag_start_pos[1] - event.pos[1])
            if math.hypot(force_vector[0], force_vector[1]) > 5:
                self.selected_plate.apply_force_and_torque(force_vector, self.click_offset)
                self.game_state = "PLATES_MOVING"
            self.selected_plate, self.drag_start_pos, self.click_offset = None, None, None

    ### 修正点2: 衝突時の回転物理を改善 ###
    def resolve_collisions(self):
        for i in range(len(self.plates)):
            for j in range(i + 1, len(self.plates)):
                p1, p2 = self.plates[i], self.plates[j]
                is_colliding, mtv = self.check_collision_sat(p1, p2)
                
                if is_colliding:
                    # 1. 位置補正
                    p1.x -= mtv.x / 2
                    p1.y -= mtv.y / 2
                    p2.x += mtv.x / 2
                    p2.y += mtv.y / 2
                    p1.update_corners()
                    p2.update_corners()

                    # 2. 衝突応答
                    n = mtv.normalize()
                    rv = pygame.math.Vector2(p2.vx - p1.vx, p2.vy - p1.vy)
                    vel_normal = rv.dot(n)
                    
                    if vel_normal > 0: continue
                        
                    # 2a. 線形衝動 (移動)
                    impulse = -(1 + COLLISION_ELASTICITY) * vel_normal / 2
                    p1.vx -= impulse * n.x
                    p1.vy -= impulse * n.y
                    p2.vx += impulse * n.x
                    p2.vy += impulse * n.y

                    # 2b. 角運動量 (回転)
                    # 中心間ベクトルと衝突ベクトルのクロス積でトルクを計算
                    r_vec = pygame.math.Vector2(p2.x - p1.x, p2.y - p1.y)
                    impulse_vec = pygame.math.Vector2(impulse * n.x, impulse * n.y)
                    
                    torque = r_vec.x * impulse_vec.y - r_vec.y * impulse_vec.x
                    
                    # 慣性モーメントを考慮して角速度を更新
                    p1.angular_velocity += torque / PLATE_INERTIA * COLLISION_TORQUE_FACTOR
                    p2.angular_velocity -= torque / PLATE_INERTIA * COLLISION_TORQUE_FACTOR

    def check_collision_sat(self, p1, p2):
        axes = self.get_axes(p1) + self.get_axes(p2)
        min_overlap, mtv_axis = float('inf'), None
        for axis in axes:
            proj1, proj2 = self.project(p1, axis), self.project(p2, axis)
            overlap = proj1[1] - proj2[0] if proj1[0] < proj2[0] else proj2[1] - proj1[0]
            if overlap < 0: return False, None
            if overlap < min_overlap:
                min_overlap, mtv_axis = overlap, axis
        mtv = mtv_axis * min_overlap
        if pygame.math.Vector2(p2.x - p1.x, p2.y - p1.y).dot(mtv) < 0:
            mtv = -mtv
        return True, mtv

    def get_axes(self, plate):
        axes = []
        for i in range(len(plate.corners)):
            edge = plate.corners[(i + 1) % len(plate.corners)] - plate.corners[i]
            axes.append(pygame.math.Vector2(-edge.y, edge.x).normalize())
        return axes

    def project(self, plate, axis):
        min_proj, max_proj = float('inf'), float('-inf')
        for corner in plate.corners:
            projection = corner.dot(axis)
            min_proj, max_proj = min(min_proj, projection), max(max_proj, projection)
        return min_proj, max_proj

    def update_game_state(self):
        if self.game_state == "PLATES_MOVING":
            for _ in range(2):
                for plate in self.plates: plate.update()
                self.resolve_collisions()
            if not any(p.is_moving() for p in self.plates):
                scored_this_turn = self.evaluate_turn_end()
                self.check_game_over()
                if self.game_state != "GAME_OVER":
                    if scored_this_turn: self._setup_board()
                    self.switch_player()
                    self.game_state = "PLAYER_TURN"

    def evaluate_turn_end(self):
        board_x_start, board_y_start = (SCREEN_WIDTH - BOARD_WIDTH) / 2, (SCREEN_HEIGHT - BOARD_HEIGHT) / 2
        board_x_end, board_y_end = board_x_start + BOARD_WIDTH, board_y_start + BOARD_HEIGHT
        out_of_bounds_plates, plates_to_keep = [], []
        for plate in self.plates:
            is_out, out_info = False, {'plate': plate, 'side': None}
            if plate.rect.bottom < board_y_start or plate.rect.top > board_y_end:
                is_out, out_info['side'] = True, 'top_bottom'
            elif plate.rect.left >= board_x_end:
                is_out, out_info['side'] = True, 'right'
            elif plate.rect.right <= board_x_start:
                is_out, out_info['side'] = True, 'left'
            if is_out: out_of_bounds_plates.append(out_info)
            else: plates_to_keep.append(plate)
        if not out_of_bounds_plates: return False
        opponent = 2 if self.current_player == 1 else 1
        if any(info['plate'].owner == opponent for info in out_of_bounds_plates):
            self.scores[opponent] += 1
        else:
            goal_p1 = self.current_player == 1 and any(i['plate'].owner == 1 and i['side'] == 'right' for i in out_of_bounds_plates)
            goal_p2 = self.current_player == 2 and any(i['plate'].owner == 2 and i['side'] == 'left' for i in out_of_bounds_plates)
            if goal_p1 or goal_p2: self.scores[self.current_player] += 2
            else: self.scores[opponent] += 1
        self.plates = plates_to_keep
        return True

    def switch_player(self):
        self.current_player = 2 if self.current_player == 1 else 1

    def check_game_over(self):
        if any(s >= WINNING_SCORE for s in self.scores.values()):
            self.game_state = "GAME_OVER"

    def draw(self):
        self.screen.blit(self.background_image, (0, 0))
        board_rect = self.board_image.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
        self.screen.blit(self.board_image, board_rect)
        pygame.draw.line(self.screen, WHITE, (SCREEN_WIDTH / 2, board_rect.top), (SCREEN_WIDTH / 2, board_rect.bottom), 2)
        pygame.draw.line(self.screen, PLAYER_2_COLOR, (board_rect.left, board_rect.top), (board_rect.left, board_rect.bottom), 8)
        pygame.draw.line(self.screen, PLAYER_1_COLOR, (board_rect.right, board_rect.top), (board_rect.right, board_rect.bottom), 8)
        for plate in self.plates:
            plate.draw(self.screen)
        if self.selected_plate and self.drag_start_pos:
            pygame.draw.line(self.screen, WHITE, self.drag_start_pos, pygame.mouse.get_pos(), 2)
        self.draw_ui()
        if self.game_state == "GAME_OVER":
            self.draw_game_over()
        pygame.display.flip()

    def draw_ui(self):
        p1_score_text = self.font.render(f"P1: {self.scores[1]}", True, PLAYER_1_COLOR)
        p2_score_text = self.font.render(f"P2: {self.scores[2]}", True, PLAYER_2_COLOR)
        self.screen.blit(p1_score_text, (20, 20))
        self.screen.blit(p2_score_text, (SCREEN_WIDTH - p2_score_text.get_width() - 20, 20))
        if self.game_state != "GAME_OVER":
            turn_color = PLAYER_1_COLOR if self.current_player == 1 else PLAYER_2_COLOR
            turn_text_str = f"Player {self.current_player}'s Turn" if self.game_state == "PLAYER_TURN" else "Moving..."
            turn_text = self.font.render(turn_text_str, True, turn_color)
            text_rect = turn_text.get_rect(center=(SCREEN_WIDTH / 2, 30))
            self.screen.blit(turn_text, text_rect)

    def draw_game_over(self):
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        winner = 1 if self.scores[1] >= WINNING_SCORE else 2
        winner_color = PLAYER_1_COLOR if winner == 1 else PLAYER_2_COLOR
        game_over_text = self.font.render("GAME OVER", True, WHITE)
        winner_text = self.font.render(f"Player {winner} Wins!", True, winner_color)
        restart_text = self.small_font.render("Click to Restart", True, WHITE)
        for text, v_offset in [(game_over_text, -50), (winner_text, 0), (restart_text, 50)]:
            rect = text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 + v_offset))
            self.screen.blit(text, rect)

    def run(self):
        while True:
            self.handle_events()
            self.update_game_state()
            self.draw()
            self.clock.tick(60)

if __name__ == '__main__':
    game = SosGame()
    game.run()