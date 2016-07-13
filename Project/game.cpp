#include "template.h"
#include "raytracer.h"

vec3 position;

// Rendering targets.
Camera     camera;
Raytracer  raytracer;
Rasterizer rasterizer;
bool       isRaytracer = true;

// -----------------------------------------------------------
// Initialize the game
// -----------------------------------------------------------
void Game::Init()
{
    raytracer.Init(screen);
    rasterizer.Init(screen);
    SGNode* t_potter = rasterizer.scene->Add("assets/maze.obj");
    t_potter->SetPosition(vec3(0, 0, 0));
    for (int i = 0; i < rasterizer.scene->meshList.size(); ++i)
    {
        MeshCollider* mc = new MeshCollider( vec3( 0, 0, 0 ),
                                             vec4( 1               ,
                                                   0.5f + i * 0.5f ,
                                                   0.5f + i * 0.5f ,
                                                   0.5f + i * 0.5f ),
                                             0.0f, 0.0f, 1.0f,
                                             rasterizer.scene->meshList[i]);

        //raytracer.m_Objects.push_back(mc);
    }
    position = vec3(0, 0, 0);
	raytracer.BuildBVH(rasterizer.scene->meshList);
}

// -----------------------------------------------------------
// Input handling
// -----------------------------------------------------------
bool pressed = false;
void Game::HandleInput( float dt )
{
    // Optimal code for lazy peoples.
    if (GetAsyncKeyState(VK_SPACE)) 
    { 
        if (!pressed)
        {
            isRaytracer = !isRaytracer;
            pressed = true;
        }
    }
    else 
    {
        pressed = false;
    }
    
	if (GetAsyncKeyState('W')){ position += camera.GetForward() * dt * 10.0f; raytracer.curLine = 0; }
	if (GetAsyncKeyState('S')){ position -= camera.GetForward() * dt * 10.0f; raytracer.curLine = 0; }
    if (GetAsyncKeyState('D')){ position += camera.GetRight()   * dt * 10.0f; raytracer.curLine = 0; }
	if (GetAsyncKeyState('A')){ position -= camera.GetRight()   * dt * 10.0f; raytracer.curLine = 0; }
	if (GetAsyncKeyState('R')){ position += camera.GetUp()      * dt * 10.0f; raytracer.curLine = 0; }
    if (GetAsyncKeyState('F')){ position -= camera.GetUp()      * dt * 10.0f; raytracer.curLine = 0; }
    camera.SetPosition(position);
    if (GetAsyncKeyState(VK_RIGHT)) { camera.LookAt(camera.GetPosition() + camera.GetForward() + dt * 2.0f * camera.GetRight()); raytracer.curLine = 0; }
    if (GetAsyncKeyState(VK_LEFT))  { camera.LookAt(camera.GetPosition() + camera.GetForward() - dt * 2.0f * camera.GetRight()); raytracer.curLine = 0; }
    if (GetAsyncKeyState(VK_DOWN))  { camera.LookAt(camera.GetPosition() + camera.GetForward() + dt * 2.0f * camera.GetUp());    raytracer.curLine = 0; }
    if (GetAsyncKeyState(VK_UP))    { camera.LookAt(camera.GetPosition() + camera.GetForward() - dt * 2.0f * camera.GetUp());    raytracer.curLine = 0; }

    if (GetAsyncKeyState('O')) { raytracer.m_Lights[0]->m_Pos  += vec3(0, dt, 0); raytracer.curLine = 0; }
    if (GetAsyncKeyState('P')) { raytracer.m_Lights[0]->m_Pos  -= vec3(0, dt, 0); raytracer.curLine = 0; }
    if (GetAsyncKeyState('U')) { raytracer.m_Objects[0]->m_Pos += vec3(0, dt, 0); raytracer.curLine = 0; }
    if (GetAsyncKeyState('I')) { raytracer.m_Objects[0]->m_Pos -= vec3(0, dt, 0); raytracer.curLine = 0; }
	if (GetAsyncKeyState('T')) { raytracer.traverseDepth = !raytracer.traverseDepth; }
	
}

// -----------------------------------------------------------
// Main game tick function
// -----------------------------------------------------------
int   frames          = 0;    // Frames since last refresh.
int   framesPerSecond = 0;    // FPS display value.
float frameTime       = 0.0f; // Frame time since last refresh.
float frameDuration   = 0.0f; // Frame duration display value.
void Game::Tick( float dt )
{
	HandleInput(dt);
	screen->Clear( 0 );

    // Render scene geometry.
    if (isRaytracer)
        raytracer.Render(camera);
    else
        rasterizer.Render(camera);

    // Re-calculate FPS count each second.
    frames++;
    frameTime += dt;
    if (frameTime >= 1.0f)
    {
        // Update display values.
        framesPerSecond = frames;
        frameDuration   = frameTime / (float)frames;
        // Reset counters.
        frames = 0;
        frameTime = 0.0f;
    }
    // Draw FPS to screen buffer.
    char c[512];
    sprintf_s(c, 512, "Frametime : %f", frameDuration);
    screen->Print(c, 2, 2 , 0xff00ff);
    sprintf_s(c, 512, "FPS       : %i", framesPerSecond);
    screen->Print(c, 2, 12, 0xff00ff);
}