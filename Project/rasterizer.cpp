#include "template.h"


// -----------------------------------------------------------
// Tiny functions
// -----------------------------------------------------------
Rasterizer::~Rasterizer() { delete scene; }

// -----------------------------------------------------------
// static data for the rasterizer
// -----------------------------------------------------------
float* Rasterizer::zbuffer;
vec4 Rasterizer::frustum[5];


// -----------------------------------------------------------
// Rasterizer::Init
// initialization of the rasterizer
// input: surface to draw to
// -----------------------------------------------------------
void Rasterizer::Init( Surface* screen )
{
	// setup outline tables & zbuffer
	Mesh::xleft = new float[SCRHEIGHT], Mesh::xright = new float[SCRHEIGHT];
	Mesh::uleft = new float[SCRHEIGHT], Mesh::uright = new float[SCRHEIGHT];
	Mesh::vleft = new float[SCRHEIGHT], Mesh::vright = new float[SCRHEIGHT];
	Mesh::zleft = new float[SCRHEIGHT], Mesh::zright = new float[SCRHEIGHT];
	for( int y = 0; y < SCRHEIGHT; y++ ) Mesh::xleft[y] = SCRWIDTH - 1, Mesh::xright[y] = 0;
	zbuffer = new float[SCRWIDTH * SCRHEIGHT];
	// calculate view frustum planes
	float C = -1.0f, x1 = 0.5f, x2 = SCRWIDTH - 1.5f, y1 = 0.5f, y2 = SCRHEIGHT - 1.5f;
	vec3 p0 = { 0, 0, 0 };
	vec3 p1 = { ((x1 - SCRWIDTH / 2) * C) / SCRWIDTH, ((y1 - SCRHEIGHT / 2) * C) / SCRWIDTH, 1.0f };
	vec3 p2 = { ((x2 - SCRWIDTH / 2) * C) / SCRWIDTH, ((y1 - SCRHEIGHT / 2) * C) / SCRWIDTH, 1.0f };
	vec3 p3 = { ((x2 - SCRWIDTH / 2) * C) / SCRWIDTH, ((y2 - SCRHEIGHT / 2) * C) / SCRWIDTH, 1.0f };
	vec3 p4 = { ((x1 - SCRWIDTH / 2) * C) / SCRWIDTH, ((y2 - SCRHEIGHT / 2) * C) / SCRWIDTH, 1.0f };
	frustum[0] = { 0, 0, -1, 0.2f };
	frustum[1] = vec4( normalize( cross( p1 - p0, p4 - p1 ) ), 0 ); // left plane
	frustum[2] = vec4( normalize( cross( p2 - p0, p1 - p2 ) ), 0 ); // top plane
	frustum[3] = vec4( normalize( cross( p3 - p0, p2 - p3 ) ), 0 ); // right plane
	frustum[4] = vec4( normalize( cross( p4 - p0, p3 - p4 ) ), 0 ); // bottom plane
	// store screen pointer
	Mesh::screen = screen;
	// initialize scene
	(scene = new Scene())->root = new SGNode();
}

// -----------------------------------------------------------
// Rasterizer::Render
// render the scene
// input: camera to render with
// -----------------------------------------------------------
void Rasterizer::Render( Camera& camera )
{
	memset( zbuffer, 0, SCRWIDTH * SCRHEIGHT * sizeof( float ) );

    RenderNode(inverse(camera.transform), scene->root);
}

// -----------------------------------------------------------
// SGNode::Render
// recursive rendering of a scene graph node and its child nodes
// input: (inverse) camera transform
// -----------------------------------------------------------
void Rasterizer::RenderNode(mat4& transform, SGNode* node)
{
    mat4 M = transform * node->localTransform;
    if (node->GetType() == SGNode::SG_MESH)
        RenderMesh(M, (Mesh*)node);
    for (uint i = 0; i < node->child.size(); i++)
        RenderNode(M, node->child[i]);
}

// -----------------------------------------------------------
// Mesh render function
// input: final matrix for scene graph node
// renders a mesh using software rasterization.
// stages:
// 1. mesh culling: checks the mesh against the view frustum
// 2. vertex transform: calculates world space coordinates
// 3. triangle rendering loop. substages:
//    a) backface culling
//    b) clipping (Sutherland-Hodgeman)
//    c) shading (using pre-scaled palettes for speed)
//    d) projection: world-space to 2D screen-space
//    e) span construction
//    f) span filling
// -----------------------------------------------------------
void Rasterizer::RenderMesh( mat4& transform, Mesh* mesh )
{
    // cull mesh
    vec3 c[8];
    for (int i = 0; i < 8; i++) c[i] = vec3(transform * vec4(mesh->bounds[i & 1].x, mesh->bounds[(i >> 1) & 1].y, mesh->bounds[i >> 2].z, 1));
    for (int i, p = 0; p < 5; p++)
    {
        for (i = 0; i < 8; i++) if ((dot(vec3(Rasterizer::frustum[p]), c[i]) - Rasterizer::frustum[p].w) > 0) break;
        if (i == 8) return;
    }
    // transform vertices
    for (int i = 0; i < mesh->verts; i++) mesh->tpos[i] = vec3(transform * vec4(mesh->pos[i], 1));
    // draw triangles
    if (!mesh->material->texture) return; // for now: texture required.
    unsigned char* src = mesh->material->texture->pixels->GetBuffer();
    float* zbuffer = Rasterizer::zbuffer, f;
    const float tw = (float)mesh->material->texture->pixels->GetWidth();
    const float th = (float)mesh->material->texture->pixels->GetHeight();
    const int umask = (int)tw - 1, vmask = (int)th - 1;
    mat3 transform3x3 = mat3(transform);
    for (int i = 0; i < mesh->tris; i++)
    {
        // cull triangle
        vec3 Nt = transform3x3 * mesh->N[i];
        if (dot(mesh->tpos[mesh->tri[i * 3 + 0]], Nt) > 0) continue;
        // clip
        vec3 cpos[2][8], *pos;
        vec2 cuv[2][8], *tuv;
        int nin = 3, nout = 0, from = 0, to = 1, miny = SCRHEIGHT - 1, maxy = 0, h;
        for (int v = 0; v < 3; v++) cpos[0][v] = mesh->tpos[mesh->tri[i * 3 + v]], cuv[0][v] = mesh->uv[mesh->tri[i * 3 + v]];
        for (int p = 0; p < 2; p++, from = 1 - from, to = 1 - to, nin = nout, nout = 0) for (int v = 0; v < nin; v++)
        {
            const vec3 A = cpos[from][v], B = cpos[from][(v + 1) % nin];
            const vec2 Auv = cuv[from][v], Buv = cuv[from][(v + 1) % nin];
            const vec4 plane = Rasterizer::frustum[p];
            const float t1 = dot(vec3(plane), A) - plane.w, t2 = dot(vec3(plane), B) - plane.w;
            if ((t1 < 0) && (t2 >= 0))
                f = t1 / (t1 - t2),
                cuv[to][nout] = Auv + f * (Buv - Auv), cpos[to][nout++] = A + f * (B - A),
                cuv[to][nout] = Buv, cpos[to][nout++] = B;
            else if ((t1 >= 0) && (t2 >= 0)) cuv[to][nout] = Buv, cpos[to][nout++] = B;
            else if ((t1 >= 0) && (t2 < 0))
                f = t1 / (t1 - t2),
                cuv[to][nout] = Auv + f * (Buv - Auv), cpos[to][nout++] = A + f * (B - A);
        }
        if (nin == 0) continue;
        // shade
        Pixel* pal = mesh->material->texture->pixels->GetPalette((int)(max(0.0f, Nt.z) * (PALETTE_LEVELS - 1)));
        // project
        pos = cpos[from], tuv = cuv[from];
        for (int v = 0; v < nin; v++)
            pos[v].x = ((pos[v].x * SCRWIDTH) / -pos[v].z) + SCRWIDTH / 2,
            pos[v].y = ((pos[v].y * SCRWIDTH) / pos[v].z) + SCRHEIGHT / 2;
        // draw
        for (int j = 0; j < nin; j++)
        {
            int vert0 = j, vert1 = (j + 1) % nin;
            if (pos[vert0].y > pos[vert1].y) h = vert0, vert0 = vert1, vert1 = h;
            const float y0 = pos[vert0].y, y1 = pos[vert1].y, rydiff = 1.0f / (y1 - y0);
            if ((y0 == y1) || (y0 >= SCRHEIGHT) || (y1 < 1)) continue;
            const int iy0 = max(1, (int)y0 + 1), iy1 = min(SCRHEIGHT - 2, (int)y1);
            float x0 = pos[vert0].x, dx = (pos[vert1].x - x0) * rydiff;
            float z0 = 1.0f / pos[vert0].z, z1 = 1.0f / pos[vert1].z, dz = (z1 - z0) * rydiff;
            float u0 = tuv[vert0].x * z0, du = (tuv[vert1].x * z1 - u0) * rydiff;
            float v0 = tuv[vert0].y * z0, dv = (tuv[vert1].y * z1 - v0) * rydiff;
            const float f = (float)iy0 - y0;
            x0 += dx * f, u0 += du * f, v0 += dv * f, z0 += dz * f;
            for (int y = iy0; y <= iy1; y++)
            {
                if (x0 < mesh->xleft[y]) mesh->xleft[y] = x0, mesh->uleft[y] = u0, mesh->vleft[y] = v0, mesh->zleft[y] = z0;
                if (x0 > mesh->xright[y]) mesh->xright[y] = x0, mesh->uright[y] = u0, mesh->vright[y] = v0, mesh->zright[y] = z0;
                x0 += dx, u0 += du, v0 += dv, z0 += dz;
            }
            miny = min(miny, iy0), maxy = max(maxy, iy1);
        }
        for (int y = miny; y <= maxy; mesh->xleft[y] = SCRWIDTH - 1, mesh->xright[y++] = 0)
        {
            float x0 = mesh->xleft[y], x1 = mesh->xright[y], rxdiff = 1.0f / (x1 - x0);
            float u0 = mesh->uleft[y], du = (mesh->uright[y] - u0) * rxdiff;
            float v0 = mesh->vleft[y], dv = (mesh->vright[y] - v0) * rxdiff;
            float z0 = mesh->zleft[y], dz = (mesh->zright[y] - z0) * rxdiff;
            const int ix0 = (int)x0 + 1, ix1 = min(SCRWIDTH - 2, (int)x1);
            const float f = (float)ix0 - x0;
            u0 += f * du, v0 += f * dv, z0 += f * dz;
            Pixel* dest = mesh->screen->GetBuffer() + y * mesh->screen->GetWidth();
            float* zbuf = zbuffer + y * SCRWIDTH;
            for (int x = ix0; x <= ix1; x++, u0 += du, v0 += dv, z0 += dz) // plot span
            {
                if (z0 >= zbuf[x]) continue;
                const float z = 1.0f / z0;
                const int u = (int)(u0 * z * tw) & umask, v = (int)(v0 * z * th) & vmask;
                dest[x] = pal[src[u + v * (umask + 1)]], zbuf[x] = z0;
            }
        }
    }
}