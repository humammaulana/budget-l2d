# Budget L2D
Real time face tracking to animate virtual avatar movement and expression using **Mediapipe** and **Pygame**.
## About Project
Because I can't afford Live2D to animate my virtual avatar, I make *"my own L2D"* to animate simple movement of my avatar using Mediapipe for face tracking and Pygame to render the avatar.
## How it Works
The program uses **parallax effect** to make illusion of depth by layering images and make the movements different on each layer.
Movement tracked by extracting position of key face landmarks from Mediapipe *(corners of eyes, iris, edges of face, etc)* then calculate position of each feature respective to others. The calculated numbers then used as render coordinate of the respective object.
## Limitations
1. **Not packaged for end-users**: This project is currently intended as personal use and learning experiment.
2. **Performance**: The program is working as intended but not yet optimized due to redundant calculations and/or unoptimized logic. Future iterations may improve structure and efficiency.
