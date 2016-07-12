#include "template.h"
#include "raytracer.h"

vec3 position;

// Rendering targets.
Camera     camera;
Raytracer  raytracer;
Rasterizer rasterizer;
bool       isRaytracer = false;

// -----------------------------------------------------------
// Initialize the game
// -----------------------------------------------------------
void Game::Init()
{
    raytracer.Init(screen);
    rasterizer.Init(screen);
    SGNode* t_potter = rasterizer.scene->Add("assets/pacman_close.obj");
    t_potter->SetPosition(vec3(0, 0, 0));
    for (int i = 0; i < rasterizer.scene->meshList.size(); ++i)
    {
        raytracer.objects.push_back(new MeshCollider(vec3(0), vec4(1, 0.5f + i * 0.5f, 0.5f + i * 0.5f, 0.5f + i * 0.5f), 0.5f, 0.0f, rasterizer.scene->meshList[i]));
    }
    position = vec3(-5, 0, 0);
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
    
	if (GetAsyncKeyState('W')){ position += camera.GetForward() * dt * 40.0f; raytracer.curLine = 0; }
	if (GetAsyncKeyState('S')){ position -= camera.GetForward() * dt * 40.0f; raytracer.curLine = 0; }
    if (GetAsyncKeyState('D')){ position += camera.GetRight()   * dt * 40.0f; raytracer.curLine = 0; }
	if (GetAsyncKeyState('A')){ position -= camera.GetRight()   * dt * 40.0f; raytracer.curLine = 0; }
	if (GetAsyncKeyState('R')){ position += camera.GetUp()      * dt * 40.0f; raytracer.curLine = 0; }
    if (GetAsyncKeyState('F')){ position -= camera.GetUp()      * dt * 40.0f; raytracer.curLine = 0; }
    camera.SetPosition(position);
    if (GetAsyncKeyState(VK_RIGHT)) { camera.LookAt(camera.GetPosition() + camera.GetForward() + dt * 4.0f * camera.GetRight()); raytracer.curLine = 0; }
    if (GetAsyncKeyState(VK_LEFT))  { camera.LookAt(camera.GetPosition() + camera.GetForward() - dt * 4.0f * camera.GetRight()); raytracer.curLine = 0; }
    if (GetAsyncKeyState(VK_DOWN))  { camera.LookAt(camera.GetPosition() + camera.GetForward() + dt * 4.0f * camera.GetUp());    raytracer.curLine = 0; }
    if (GetAsyncKeyState(VK_UP))    { camera.LookAt(camera.GetPosition() + camera.GetForward() - dt * 4.0f * camera.GetUp());    raytracer.curLine = 0; }

	if (GetAsyncKeyState('O')) 
	{ 
		raytracer.bvh->test = true;
	}

	if (GetAsyncKeyState('P'))
	{
		raytracer.bvh->test = false;
	}
}

// -----------------------------------------------------------
// Main game tick function
// -----------------------------------------------------------
void Game::Tick( float dt )
{
	HandleInput(dt);
	screen->Clear( 0 );
	screen->Print( "hello world", 2, 2, 0xffffff );
	screen->Line( 2, 10, 50, 10, 0xff0000 );

    if (isRaytracer)
        raytracer.Render(camera);
    else
        rasterizer.Render(camera);
}