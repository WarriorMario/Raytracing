#pragma once

namespace Tmpl8
{

// -----------------------------------------------------------
// Rasterizer class
// rasterizer
// implements a basic, but fast & accurate software rasterizer,
// with the following features:
// - frustum culling (per mesh)
// - backface culling (per tri)
// - full frustum clipping, including near plane
// - perspective correct texture mapping
// - sub-pixel and sub-texel accuracy
// - z-buffering
// - basic shading (n dot l for an imaginary light source)
// - fast OBJ file loading with render state oriented mesh breakdown
// this rasterizer has been designed for educational purposes
// and is intentionally small and bare bones.
// -----------------------------------------------------------
class Rasterizer
{
public:
	// constructor / destructor
	Rasterizer() : scene( 0 ) {}
	~Rasterizer();
	// methods
	void Init( Surface* screen );
	void Render( Camera& camera );
    void RenderNode( mat4& transform, SGNode* node );
    void RenderMesh( mat4& transform, Mesh* mesh );
	// data members
	Scene* scene;	
	static float* zbuffer;
	static vec4 frustum[5];
};

}; // namespace Tmpl8