import shared
import face_detect

face_detect.start()


import pygame
import sys

pygame.init()

screen_width, screen_height = 640, 480
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Budget L2D")

head_img = pygame.image.load('images/avatar/head.png').convert_alpha()
head_rect = head_img.get_rect()

l_eye_img = pygame.image.load('images/avatar/l_eye.png').convert_alpha()
l_eye_img2 = pygame.image.load('images/avatar/l_eye2.png').convert_alpha()
l_eye_img3 = pygame.image.load('images/avatar/l_eye3.png').convert_alpha()

r_eye_img = pygame.image.load('images/avatar/r_eye.png').convert_alpha()
r_eye_img2 = pygame.image.load('images/avatar/r_eye2.png').convert_alpha()
r_eye_img3 = pygame.image.load('images/avatar/r_eye3.png').convert_alpha()

l_iris_img = pygame.image.load('images/avatar/l_iris.png').convert_alpha()
r_iris_img = pygame.image.load('images/avatar/r_iris.png').convert_alpha()
l_iris_rect = l_iris_img.get_rect()
r_iris_rect = r_iris_img.get_rect()

l_eye_mask = pygame.Surface(l_eye_img.get_size(), pygame.SRCALPHA)
r_eye_mask = pygame.Surface(r_eye_img.get_size(), pygame.SRCALPHA)

clock = pygame.time.Clock()

while shared.running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            shared.running = False
            pygame.quit()
            sys.exit()
    
    screen.fill((0, 0, 0))
    l_eye_mask.fill((0, 0, 0, 0))
    r_eye_mask.fill((0, 0, 0, 0))

    head_pos = (screen_width//2 - head_rect.width//2 - shared.delta_x * 50,
                screen_height//2 - head_rect.height//2 - shared.delta_y * 300)

# ==== LEFT EYE
    if shared.l_eye_pct > 0.51:
        l_eye_img_render = l_eye_img
    elif shared.l_eye_pct > 0.28:
        l_eye_img_render = l_eye_img2
    else:
        l_eye_img_render = l_eye_img3
    l_eye_rect = l_eye_img_render.get_rect()
    l_eye_mask = pygame.Surface(l_eye_img_render.get_size(), pygame.SRCALPHA)
    
    l_eye_pos = (head_rect.width//4 - l_eye_rect.width//2 + head_pos[0] - shared.delta_x*70,
                 head_rect.height//2 - l_eye_rect.height//2 + head_pos[1] - shared.delta_y*400)
    
    l_iris_pos = (
        (l_eye_rect.width//2 + shared.l_iris_x * (l_eye_rect.width//2) - l_iris_rect.width//2),
        (l_eye_rect.height//2 + shared.l_iris_y * (l_eye_rect.height//2) - l_iris_rect.height//2)
    )
    l_eye_mask.blit(l_iris_img, l_iris_pos)
    l_eye_mask.blit(l_eye_img_render, (0,0), special_flags=pygame.BLEND_RGBA_MULT)

# === RIGHT EYE
    if shared.r_eye_pct > 0.51:
        r_eye_img_render = r_eye_img
    elif shared.r_eye_pct > 0.28:
        r_eye_img_render = r_eye_img2
    else:
        r_eye_img_render = r_eye_img3
    r_eye_rect = r_eye_img_render.get_rect()
    r_eye_mask = pygame.Surface(r_eye_img_render.get_size(), pygame.SRCALPHA)

    r_eye_pos = (3*head_rect.width//4 - r_eye_rect.width//2 + head_pos[0] - shared.delta_x*70,
                head_rect.height//2 - r_eye_rect.height//2 + head_pos[1] - shared.delta_y*400)

    r_iris_pos = (
        (r_eye_rect.width//2 - shared.r_iris_x * (r_eye_rect.width//2) - r_iris_rect.width//2),
        (r_eye_rect.height//2 + shared.r_iris_y * (r_eye_rect.height//2) - r_iris_rect.height//2)
    )
    r_eye_mask.blit(r_iris_img, r_iris_pos)
    r_eye_mask.blit(r_eye_img_render, (0,0), special_flags=pygame.BLEND_RGBA_MULT)

    # === Render
    screen.blit(head_img, head_pos)
    screen.blit(l_eye_img_render, l_eye_pos)
    screen.blit(l_eye_mask, l_eye_pos)
    screen.blit(r_eye_img_render, r_eye_pos)
    screen.blit(r_eye_mask, r_eye_pos)

# ==== CLOSE LOOP
    pygame.display.flip()
    clock.tick(30)


# try:
#     while shared.running:
#         print(f"head_h: {head_h:.4f}; head_w: {head_w:.4f}; eye: {l_eye_h:.4f}; mouth: {mouth_h:.4f}; l_eye_w: {l_eye_w:.4f}; r_eye_w: {r_eye_w:.4f}")
# except KeyboardInterrupt:
#     shared.running = False