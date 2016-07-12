#include "template.h"

//*************************************************************************

#define BACKGROUND vec4(1.0f, 0.3f, 0.3f, 0.3f)
#define AMBIENT vec4(1.0f,0.5f,0.5f,0.5f)
#define EPSILON 1e-4f

#define MAXREFLPASSES 5
#define MAXREFRPASSES 5
#define FOV 1.8f

#define MISS    0
#define HIT     1
#define INPRIM -1

//*************************************************************************

using namespace Tmpl8;

//*************************************************************************

// ============================================
// Utility functions.
// ============================================
Pixel VecToPixel(vec4 color)
{
	Pixel result = 0;
	result += (unsigned int)(color.x * 255) << 24;
	result += (unsigned int)(color.y * 255) << 16;
	result += (unsigned int)(color.z * 255) << 8;
	result += (unsigned int)(color.w * 255);
	return result;
}
vec4 PixelToVec(Pixel color)
{
	vec4 result = vec4(0, 0, 0, 0);
	result.x = ((float)((color & 0xff000000) >> 24)) / 255.0f;
	result.y = ((float)((color & 0x00ff0000) >> 16)) / 255.0f;
	result.z = ((float)((color & 0x0000ff00) >> 8)) / 255.0f;
	result.w = ((float)((color & 0x000000ff))) / 255.0f;
	return result;
}
vec4 ScaleColor(unsigned int a_Scale, vec4 color)
{
	Pixel c = VecToPixel(color);
	unsigned int rb = (((c & (REDMASK | BLUEMASK)) * a_Scale) >> 5) & (REDMASK | BLUEMASK);
	unsigned int g = (((c & GREENMASK) * a_Scale) >> 5) & GREENMASK;
	return PixelToVec(rb + g);
}
Pixel RandColor()
{
	int r = (rand() % 255) << 16;
	int g = (rand() % 255) << 8;
	int b = rand() % 255;
	return (0xff << 24) | r | g | b;
}
vec3 Reflect(vec3 I, vec3 N)
{
	/* Calculate reflection direction R
	*        N
	*     \  |  /
	*    I \ | / R
	* ______\|/______
	*
	*/

	// Calculate length along normal.
	float nl = dot(I, N);
	// Calculate projected vector in normal direction.
	vec3 projN = nl * N;
	// Create tangent vector by eqaulling with hit direction.
	vec3 tan = I - projN;
	// Add once more to get R.
	return (tan - projN);
}
bool RayTriangleIntersect(Ray & ray,
    const vec3  &v0, const vec3 &v1, const vec3 &v2,
    float &u, float &v)
{
    vec3 v0v1 = v1 - v0;
    vec3 v0v2 = v2 - v0;
    vec3 pvec = cross(ray.D, v0v2);
    float det = dot(v0v1, pvec);

    // if the determinant is negative the triangle is backfacing
    // if the determinant is close to 0, the ray misses the triangle
    if (det < -EPSILON) return FALSE;
    float invDet = 1 / det;

    vec3 tvec = ray.O - v0;
    u = dot(tvec, pvec) * invDet;
    if (u < 0 || u > 1) return FALSE;

    vec3 qvec = cross(tvec, v0v1);
    v = dot(ray.D, qvec) * invDet;
    if (v < 0 || u + v > 1) return FALSE;

    float T = dot(v0v2, qvec) * invDet;

    if (T < 0.0f)
    {
        return FALSE;
    }
    else if (T < ray.t)
    {
        ray.t = T;
        return TRUE;
    }
    return FALSE;
}
bool RayTriangleHit(const Ray & ray, const vec3 &v0, const vec3 &v1, const vec3 &v2)
{
    vec3 v0v1 = v1 - v0;
    vec3 v0v2 = v2 - v0;
    vec3 pvec = cross(ray.D, v0v2);
    float det = dot(v0v1, pvec);

    // if the determinant is negative the triangle is backfacing
    // if the determinant is close to 0, the ray misses the triangle
    if (det < -EPSILON) return FALSE;
    float invDet = 1 / det;

    float u, v;
    vec3 tvec = ray.O - v0;
    u = dot(tvec, pvec) * invDet;
    if (u < 0 || u > 1) return FALSE;

    vec3 qvec = cross(tvec, v0v1);
    v = dot(ray.D, qvec) * invDet;
    if (v < 0 || u + v > 1) return FALSE;

    float T = dot(v0v2, qvec) * invDet;

    if (T < 0.0f)
    {
        return FALSE;
    }
    return TRUE;
}

//*************************************************************************

// ============================================
//                   Sphere
// ============================================
vec3 Sphere::HitNormal(const vec3& a_Pos)
{
	return (a_Pos - m_Pos) / m_Radius;
}
BOOL Sphere::Intersect(Ray& a_Ray)
{
	vec3 c = m_Pos - a_Ray.O;
	float t = dot(c, a_Ray.D);
	if (t < 0.0f) return FALSE;
	vec3 q = c - t*a_Ray.D;
	float p2 = dot(q, q);
	float r2 = m_Radius * m_Radius;
	if (p2 > r2) return FALSE;
	t -= sqrtf(r2 - p2);
	if (t < a_Ray.t)
	{
		a_Ray.t = t;
		return TRUE;
	}
	return FALSE;
}
BOOL Sphere::IsOccluded(const Ray& a_Ray)
{
	vec3 c = m_Pos - a_Ray.O;
	float t = dot(c, a_Ray.D);
	if (t < 0.0f) return FALSE;
	vec3 q = c - t*a_Ray.D;
	float p2 = dot(q, q);
	float r2 = m_Radius * m_Radius;
	if (p2 < r2) return TRUE;
	return FALSE;
}

// ============================================
//                    Plane
// ============================================
vec3 Plane::HitNormal(const vec3& a_Pos)
{
	return m_Pos;
}
BOOL Plane::Intersect(Ray& a_Ray)
{
	// t =  (N.O+dist) / N.D
	// Note: Plane normal is stored in pos.
	float dnd = dot(m_Pos, a_Ray.D);
	if (dnd == 0.0f) return FALSE;
	float t = -(dot(m_Pos, a_Ray.O) + m_Dist) / dnd;
	if (t < 0.0f) return FALSE;
	if (t < a_Ray.t)
	{
		a_Ray.t = t;
		return TRUE;
	}
	return FALSE;
}
BOOL Plane::IsOccluded(const Ray& a_Ray)
{
	// t =  (N.O+dist) / N.D
	// Note: Plane normal is stored in pos.
	float dnd = dot(m_Pos, a_Ray.D);
	if (dnd == 0) return FALSE;
	float t = -(dot(m_Pos, a_Ray.O) + m_Dist) / dnd;
	if (t < 0.0f) return FALSE;
    return TRUE;
}

// ============================================
//                Mesh collider
// ============================================
vec3 MeshCollider::HitNormal(const vec3 & a_Pos)
{
    return m_Mesh->N[index];
}
BOOL MeshCollider::IsOccluded(const Ray & a_Ray)
{
    bool hit = FALSE;
    for (int i = 0; i < m_Mesh->tris; i++)
    {
        // Get the triangle
        vec3 p0 = m_Mesh->pos[m_Mesh->tri[i * 3]];
        vec3 p1 = m_Mesh->pos[m_Mesh->tri[i * 3 + 1]];
        vec3 p2 = m_Mesh->pos[m_Mesh->tri[i * 3 + 2]];
        if (RayTriangleHit(a_Ray, p0, p1, p2))
        {
            return TRUE;
        }
    }
    return FALSE;
}
BOOL MeshCollider::Intersect(Ray& a_Ray)
{
    bool hit = FALSE;
    for (int i = 0; i < m_Mesh->tris; i++)
    {
        // Get the triangle
        vec3 p0 = m_Mesh->pos[m_Mesh->tri[i * 3]];
        vec3 p1 = m_Mesh->pos[m_Mesh->tri[i * 3 + 1]];
        vec3 p2 = m_Mesh->pos[m_Mesh->tri[i * 3 + 2]];
        float u, v;
        if (RayTriangleIntersect(a_Ray, p0, p1, p2, u, v))
        {
            vec2 uv0 = m_Mesh->uv[m_Mesh->tri[i * 3]];
            vec2 uv1 = m_Mesh->uv[m_Mesh->tri[i * 3 + 1]];
            vec2 uv2 = m_Mesh->uv[m_Mesh->tri[i * 3 + 2]];

            vec2 uvPos = uv0 + u * (uv1 - uv0) + v * (uv2 - uv0);

            Surface8 * surface = m_Mesh->material->texture->pixels;
            vec3 NT = mat3(m_Mesh->localTransform) * m_Mesh->N[i];
            Pixel* pal = surface->GetPalette(15);
            unsigned char* src = m_Mesh->material->texture->pixels->GetBuffer();

            const int tw = m_Mesh->material->texture->pixels->GetWidth();
            const int th = m_Mesh->material->texture->pixels->GetHeight();

            const int umask = (int)tw - 1, vmask = (int)th - 1;

            int xBuffer = uvPos.x * tw;
            int yBuffer = uvPos.y * th;
            m_Diffuse = PixelToVec(pal[src[yBuffer * tw + xBuffer]]);
            /////////////////////////////////////////
            // wen wi fuond kolor, m_Diffuse = kolor,
            /////////////////////////////////////////
            index = i;
            hit = TRUE;
        }
    }
    return hit;
}

// ============================================
//                  Triangle
// ============================================
vec3 Triangle::HitNormal(const vec3 & a_Pos)
{
    return normal;
}
BOOL Triangle::Intersect(Ray & a_Ray)
{
    return 0;
}
BOOL Triangle::Intersect(Ray & a_Ray, float & a_U, float & a_V)
{
    vec3 v0v1 = p1 - p0;
    vec3 v0v2 = p2 - p0;
    vec3 pvec = cross(a_Ray.D, v0v2);
    float det = dot(v0v1, pvec);
    float u, v;
    // if the determinant is negative the triangle is backfacing
    // if the determinant is close to 0, the ray misses the triangle
    if (det < -EPSILON) return FALSE;
    float invDet = 1 / det;

    vec3 tvec = a_Ray.O - p0;
    u = dot(tvec, pvec) * invDet;
    if (u < 0 || u > 1) return FALSE;

    vec3 qvec = cross(tvec, v0v1);
    v = dot(a_Ray.D, qvec) * invDet;
    if (v < 0 || u + v > 1) return FALSE;

    float T = dot(v0v2, qvec) * invDet;

    if (T < 0.0f)
    {
        return FALSE;
    }
    else if (T < a_Ray.t)
    {
        a_Ray.t = T;
        a_U = u;
        a_V = v;
        return TRUE;
    }
    return FALSE;
}
BOOL Triangle::IsOccluded(const Ray & a_Ray)
{
    return 0;
}

Color Triangle::GetColor(float a_U, float a_V)
{
    vec2 uvPos = uv0 + a_U * (uv1 - uv0) + a_V * (uv2 - uv0);

    Pixel* pal = m_Tex->pixels->GetPalette(15);
    unsigned char* src = m_Tex->pixels->GetBuffer();

    const int tw = m_Tex->pixels->GetWidth();
    const int th = m_Tex->pixels->GetHeight();

    const int umask = (int)tw - 1, vmask = (int)th - 1;

    int xBuffer = uvPos.x * tw;
    int yBuffer = uvPos.y * th;
    return PixelToVec(pal[src[yBuffer * tw + xBuffer]]);
}

//*************************************************************************

// ============================================
//                  Raytracer
// ============================================
void Raytracer::Init(Surface * screen)
{
	this->screen = screen;
	this->frameBuffer = (Pixel*)_aligned_malloc(screen->GetHeight() * screen->GetWidth() * sizeof(Pixel), 32);
	memset(frameBuffer, 0, screen->GetHeight() * screen->GetWidth() * sizeof(Pixel));
	this->curLine = 0;


    //---------------------
    //     Add lights
    //---------------------
    Light* l   = new Light();
    l->m_Pos   = vec3(0, 5, 0);
    l->m_Color = Color(1, 1, 1, 1);
    m_Lights.push_back(l);

    //---------------------
    //     Add objects
    //---------------------
    Sphere* sphere;
    sphere = new Sphere( vec3( 1, 0, 0)         ,
                         PixelToVec(0xff00ff00) ,
                         0.5f, 0.0f, 1.0f, 0.5f );
    m_Objects.push_back((Object*)sphere);
    sphere = new Sphere( vec3(-1, 0, 0)         ,
                         PixelToVec(0xffffffff) ,
                         0.0f, 0.5f, 1.5f, 0.5f );
    m_Objects.push_back((Object*)sphere);
}
BOOL Raytracer::IsOccluded(Ray & ray)
{
    BVHResult res;
	return bvh->Traverse(ray, res);
}
void Raytracer::Render(Camera & camera)
{

	vec3 p0 = vec3( camera.transform * vec4( vec3( -1,  SCRASPECT, -FOV ), 1) );
	vec3 p1 = vec3( camera.transform * vec4( vec3(  1,  SCRASPECT, -FOV ), 1) );
	vec3 p2 = vec3( camera.transform * vec4( vec3( -1, -SCRASPECT, -FOV ), 1) );
	float invHeight = 1.0f / (float)screen->GetHeight();
	float invWidth = 1.0f / (float)screen->GetWidth();

	for (int y = 0; y < screen->GetHeight(); ++y)
	{
		float v = (float)y *invHeight;
		int line = y * screen->GetPitch();
		for (int x = 0; x < screen->GetWidth(); ++x)
		{
			float u = (float)x *invWidth;
			float distance = 1.0f;
			vec3 planepos = p0 + u * (p1 - p0) + v * (p2 - p0);

            // Cast a ray into the scene.
			Ray ray = Ray(camera.GetPosition(), normalize(planepos - camera.GetPosition()), 100000000);
            int reflPasses = 0, refrPasses = 0;
			screen->GetBuffer()[x + line] = VecToPixel(GetColorFromSphere(ray, reflPasses, refrPasses, 1.0f));
		}
	}
}
void Raytracer::RenderScanlines(Camera & camera)
{
	vec3 p0 = vec3( camera.transform * vec4( vec3( -1,  SCRASPECT, -FOV ), 1) );
	vec3 p1 = vec3( camera.transform * vec4( vec3(  1,  SCRASPECT, -FOV ), 1) );
	vec3 p2 = vec3( camera.transform * vec4( vec3( -1, -SCRASPECT, -FOV ), 1) );
	float invHeight = 1.0f / (float)screen->GetHeight();
	float invWidth = 1.0f / (float)screen->GetWidth();

	int y = curLine;
	float v = (float)y *invHeight;
	int line = y * screen->GetPitch();
	for (int x = 0; x < screen->GetWidth(); ++x)
	{
		float u = (float)x *invWidth;
		float distance = 1.0f;
		vec3 planepos = p0 + u * (p1 - p0) + v * (p2 - p0);
		Ray ray = Ray(camera.GetPosition(), normalize(planepos - camera.GetPosition()), 100000000);

        int reflPasses = 0, refrPasses = 0;
		int depth = 0;
		//vec4  color = GetBVHDepth(ray,depth);
		//frameBuffer[x + line] = VecToPixel(vec4(1, (float)depth / 20, 1.0f - (float)depth / 20, 0));
		frameBuffer[x + line] = VecToPixel(GetColorFromSphere(ray, reflPasses, refrPasses, 1.0f));
		if (depth > 1)
		{
			int i = 0;
		}
	}
	memcpy(screen->GetBuffer(), frameBuffer, screen->GetHeight() * screen->GetWidth() * sizeof(Pixel));
	if (curLine < screen->GetHeight() - 1)
		curLine++;
}
Color Raytracer::GetColorFromSphere(Ray& a_Ray, int& a_ReflPass, int& a_RefrPass, float a_RIndex)
{
    // If hit was null terminate ray and return background color.
    BVHResult res;
    BOOL isHit = bvh->Traverse(a_Ray, res);
    if (!isHit) return BACKGROUND;

    // Note: ATM all objects are triangles.
    Triangle* tri = res.m_Triangle;
    
    vec3 hit = a_Ray.O + a_Ray.D * a_Ray.t;
    vec3 normal = tri->normal;
    vec4 finalColor = vec4(0.0f);
    vec4 colorP = tri->GetColor(res.m_U, res.m_V); // + res.m_Object->GetColor();

    //if (obj->isLight) return obj->GetColor();
    // Light
    float lightStrength = 1.0f - (tri->m_Refr + tri->m_Refl);
    if (lightStrength > 0.0f)
    {
        finalColor += AMBIENT * colorP * lightStrength;
        for (int i = 0; i < m_Lights.size(); ++i)
        {
            Light& light = *m_Lights[i];

            Ray lightRay = Ray(hit, normalize(light.m_Pos - hit), INFINITY);
            if (IsOccluded(lightRay) == false)
            {
                float d = dot(normal, lightRay.D);
                if (d < 0.0f)
                {
                    d = 0.0f;
                }
                finalColor += d*light.m_Color * colorP * lightStrength;
            }
        }
    }
    // Reflect.
    if (tri->m_Refl > 0.0f && a_ReflPass < MAXREFLPASSES)
    {
        a_ReflPass++;
        Ray reflRay = Ray(hit, Reflect(a_Ray.D, normal), INFINITY);
        vec4 reflColor = GetColorFromSphere(reflRay, a_ReflPass, a_RefrPass, a_RIndex);
        finalColor += reflColor * colorP * tri->m_Refl;
    }
    // Refract.
    if (tri->m_Refr > 0.0f && a_RefrPass < MAXREFRPASSES)
    {
        a_RefrPass++;

        float rindex = tri->m_RIndex;
        float n = a_RIndex / rindex;
        float cosI = -dot(normal, a_Ray.D);
        float cosT2 = 1.0f - n * n * (1.0f - cosI * cosI);
        if (cosT2 > 0.0f)
        {
            // Calculate and cast reflection ray.
            vec3 T = (n * a_Ray.D) + (n * cosI - sqrtf(cosT2)) * normal;
            Ray refrRay = Ray(hit + T * EPSILON, T, INFINITY);
            vec4 refrColor = GetColorFromSphere(refrRay, a_ReflPass, a_RefrPass, rindex);
            // Apply Beer's law.
            vec4 absorbance = colorP * 0.15f * -a_Ray.t;
            vec4 transparency = vec4(expf(absorbance.r), expf(absorbance.g), expf(absorbance.b), 1.0f);
            finalColor += refrColor * transparency;
        }
    }

	//float reflectance = m_Objects[closestIndex]->m_Refl;
	//float refractance = 0.4f;
	//float diffuse = 1.0f - reflectance;// -refractance;
	//vec4 colorR = reflColor * reflectance;
	//vec4 colorT = refrColor * refractance;

	//vec4 finalColor = lightColor;// +colorR*colorP;
	//vec4 finalColor = (AMBIENT + lightColor) * (colorR + colorP);

	return clamp(finalColor, 0.0f, 1.0f);
}
vec4 Raytracer::GetBVHDepth(Ray & ray, int& depth)
{
    return vec4(1, 1, 1, 1);
	//return bvh->TraverseDepth(ray, depth);
}
void Raytracer::BuildBVH(vector<Mesh*> meshList)
{
	bvh = new BVH(meshList);
}

//*************************************************************************

// ============================================
//                BVH utilities
// ============================================
unsigned GroupX(Triangle* triangles, unsigned int*indexArray, unsigned int start, unsigned int count, float x)
{
	int i, j, t;
	i = start;
	j = start + count - 1;
	int k = start;
	for (k = start; k < j; ++k)
	{
		Triangle triangle = triangles[indexArray[k]];
		float p = ((triangle.p0 + triangle.p1 + triangle.p2) / 3.0f).x;
		if (p <= x)
		{
			t = indexArray[i];
			indexArray[i] = indexArray[k];
			indexArray[k] = t;
			i++;
		}
		else
		{
			t = indexArray[j];
			indexArray[j] = indexArray[k];
			indexArray[k] = t;
			j--;
			k--;
		}
	}
	return k;
}
unsigned GroupY(Triangle* triangles, unsigned int*indexArray, unsigned int start, unsigned int count, float y)
{
	int i, j, t;
	i = start;
	j = start + count - 1;
	int k = start;
	for (k = start; k < j; ++k)
	{
		Triangle triangle = triangles[indexArray[k]];
		float p = ((triangle.p0 + triangle.p1 + triangle.p2) / 3.0f).y;
		if (p <= y)
		{
			t = indexArray[i];
			indexArray[i] = indexArray[k];
			indexArray[k] = t;
			i++;
		}
		else
		{
			t = indexArray[j];
			indexArray[j] = indexArray[k];
			indexArray[k] = t;
			j--;
			k--;
		}
	}
	return k;
}
unsigned GroupZ(Triangle* triangles, unsigned int*indexArray, unsigned int start, unsigned int count, float z)
{
	int i, j, t;
	i = start;
	j = start + count - 1;
	int k = start;
	for (k = start; k < j; ++k)
	{
		Triangle triangle = triangles[indexArray[k]];
		float p = ((triangle.p0 + triangle.p1 + triangle.p2) / 3.0f).z;
		if (p <= z)
		{
			t = indexArray[i];
			indexArray[i] = indexArray[k];
			indexArray[k] = t;
			i++;
		}
		else
		{
			t = indexArray[j];
			indexArray[j] = indexArray[k];
			indexArray[k] = t;
			j--;
			k--;
		}
	}
	return k;
}

void SortX(Triangle* triangles, unsigned int*indexArray, unsigned int start, unsigned int end)
{
	if ((end - start) < 2)
		return;
	unsigned int L = start;
	unsigned int R = end;
	float center = triangles[indexArray[((L + R) / 2)]].m_Pos.x;

	while (L < R)
	{
		while (triangles[indexArray[L]].m_Pos.x < center)
		{
			L++;
		}
		while (triangles[indexArray[R]].m_Pos.x > center)
		{
			R--;
		}

		if (L <= R)
		{
			unsigned int left = indexArray[L];
			indexArray[L] = indexArray[R];
			indexArray[R] = left;
			L++;
			R--;
		}
	} 

	if (start < R)
	{
		SortX(triangles, indexArray, start, R);
	}
	if (L < (end))
	{
		SortX(triangles, indexArray, L, end);
	}
}
void SortY(Triangle* triangles, unsigned int*indexArray, unsigned int start, unsigned int end)
{
	if ((end - start) < 2)
		return;
	unsigned int L = start;
	unsigned int R = end;
	float center = triangles[indexArray[((L + R) / 2)]].m_Pos.y;

	while (L < R)
	{
		while (triangles[indexArray[L]].m_Pos.y < center)
		{
			L++;
		}
		while (triangles[indexArray[R]].m_Pos.y > center)
		{
			R--;
		}

		if (L <= R)
		{
			unsigned int left = indexArray[L];
			indexArray[L] = indexArray[R];
			indexArray[R] = left;
			L++;
			R--;
		}
	} 

	if (start < R)
	{
		SortY(triangles, indexArray, start, R);
	}
	if (L < (end))
	{
		SortY(triangles, indexArray, L, end);
	}
}
void SortZ(Triangle* triangles, unsigned int*indexArray, unsigned int start, unsigned int end)
{
	if ((end - start) < 2)
		return;
	unsigned int L = start;
	unsigned int R = end;
	float center = triangles[indexArray[((L + R) / 2)]].m_Pos.z;

	while (L < R)
	{
		while (triangles[indexArray[L]].m_Pos.z < center)
		{
			L++;
		}
		while (triangles[indexArray[R]].m_Pos.z > center)
		{
			R--;
		}

		if (L <= R)
		{
			unsigned int left = indexArray[L];
			indexArray[L] = indexArray[R];
			indexArray[R] = left;
			L++;
			R--;
		}
	}

	if (start < R)
	{
		SortZ(triangles, indexArray, start, R);
	}
	if (L < (end))
	{
		SortZ(triangles, indexArray, L, end);
	}
}

bool CheckBox(vec3& bmin, vec3& bmax, vec3 O, vec3 rD, float t)
{
    vec3 tMin = (bmin - O) / rD, tMax = (bmax - O) / rD;
    vec3 t1 = min(tMin, tMax), t2 = max(tMin, tMax);
    float tNear = max(max(t1.x, t1.y), t1.z);
    float tFar = min(min(t2.x, t2.y), t2.z);
    return ((tFar > tNear) && (tNear < t) && (tFar > 0));
}


// ============================================
//                     BVH
// ============================================
BVH::BVH(vector<Mesh*> meshes)
{
	nTris = 0;
	for (int i = 0; i < meshes.size(); ++i)
	{
		nTris += meshes[i]->tris;
	}
    
	m_Triangles = (Triangle*)malloc(nTris * sizeof(Triangle));
	m_TriangleIdx = (unsigned int*)malloc(nTris * sizeof(int));
	unsigned int trIdx = 0;
	for (int i = 0; i < meshes.size(); ++i)
	{
        if (i > 0)
        {
            // floor
            for (int j = 0; j < meshes[i]->tris; ++j)
            {
                m_Triangles[trIdx] = Triangle(vec3(0), vec4(0), 0.4f, 0.0f, 1.0f, meshes[i], j);
                m_TriangleIdx[trIdx] = trIdx;
                trIdx++;
            }
        }
        else
        {
            // mazes
            for (int j = 0; j < meshes[i]->tris; ++j)
            {
                m_Triangles[trIdx] = Triangle(vec3(0), vec4(0), 0.6f, 0.3f, 1.5f, meshes[i], j);
                m_TriangleIdx[trIdx] = trIdx;
                trIdx++;
            }
        }
	}

	// check the size of BVHNode
	size_t size = sizeof(BVHNode);
	nNodes = nTris * 2 + 1;
	m_Nodes = (BVHNode*)_aligned_malloc(nNodes * sizeof(BVHNode), 128);

	BVHNode* root = &m_Nodes[0];
	poolPtr = 0;
    maxDepth = 0;
	root->firstLeft = 0;
	root->count = nTris;
	root->Subdivide(this, 0); // construct the first node
}
BOOL BVH::Traverse(Ray & a_Ray, BVHResult& a_Result)
{
	return m_Nodes[0].Traverse(this, a_Ray, a_Result);
}
BOOL BVH::TraverseDepth(Ray & a_Ray, int& depth, BVHResult& a_Result)
{
    return m_Nodes[0].TraverseDepth(this, a_Ray, depth, a_Result);
}
void BVHNode::Subdivide(BVH * bvh, int depth)
{
    bvh->maxDepth = max(bvh->maxDepth, depth);
    // Calculate bounds
    m_AABB = AABB(bvh->m_Triangles, bvh->m_TriangleIdx, firstLeft, count);

    // Quit if we do not have enough primitives
    if ((count) < bvh->primPerNode)
        return;
    // Construct two nodes
    bvh->poolPtr++;
    BVHNode* leftNode = &bvh->m_Nodes[bvh->poolPtr];
    bvh->poolPtr++;
    BVHNode* rightNode = &bvh->m_Nodes[bvh->poolPtr];

    // Partition
    // Find longest axis
    float dx, dy, dz;
    dx = m_AABB.m_Max.x - m_AABB.m_Min.x;
    dy = m_AABB.m_Max.y - m_AABB.m_Min.y;
    dz = m_AABB.m_Max.z - m_AABB.m_Min.z;
    int splitIndexX = 0;
    int splitIndexY = 0;
    int splitIndexZ = 0;
    float initCost = m_AABB.CalculateVolume()*count;
    float bestCost = m_AABB.CalculateVolume()*count;
    int bestAxis = 0;
    //if (dx > dy && dx > dz)
    {
        // Split on x
        float posStep = dx / 10.0f;
        SortX(bvh->m_Triangles, bvh->m_TriangleIdx, firstLeft, firstLeft + count - 1);
        // Find best splitpoint
        // Offset now
        float splitPoint = posStep + m_AABB.m_Min.x;
        int startIndex = firstLeft;
        int bestSplitIndex = startIndex;
        for (int i = 0; i < 8; ++i)
        {
            // Find point where we surpass the split point
            for (int iTriangle = startIndex; iTriangle < startIndex + count; ++iTriangle)
            {
                Triangle* triangle = &bvh->m_Triangles[bvh->m_TriangleIdx[iTriangle]];
                if (triangle->m_Pos.x > splitPoint)
                {
                    // Calculate new AABB
                    float leftVolume, rightVolume;
                    int leftCount = (iTriangle - startIndex);
                    if (leftCount == 0)
                    {
                        leftVolume = 0;
                    }
                    else
                    {
                        AABB left = AABB(bvh->m_Triangles, bvh->m_TriangleIdx, startIndex, leftCount);
                        leftVolume = left.CalculateVolume();
                    }

                    int rightCount = count - leftCount;
                    if (rightCount == 0)
                    {
                        rightVolume = 0;
                    }
                    else
                    {

                        AABB right = AABB(bvh->m_Triangles, bvh->m_TriangleIdx, iTriangle, rightCount);
                        rightVolume = right.CalculateVolume();
                    }
                    float cost = leftVolume * leftCount + rightVolume * rightCount;
                    if (cost < bestCost)
                    {
                        bestCost = cost;
                        // New start index
                        bestSplitIndex = iTriangle;
                        bestAxis = 0;
                    }
                    break;
                }
            }
            splitPoint += posStep;
        }
        splitIndexX = bestSplitIndex;
    }
    //else if (dy > dz)
    {
        // Split on y
        float posStep = dy / 10.0f;
        SortY(bvh->m_Triangles, bvh->m_TriangleIdx, firstLeft, firstLeft + count - 1);
        // Find best splitpoint
        // Offset now
        float splitPoint = posStep + m_AABB.m_Min.y;
        int startIndex = firstLeft;
        int bestSplitIndex = startIndex;
        for (int i = 0; i < 8; ++i)
        {
            // Find point where we surpass the split point
            for (int iTriangle = startIndex; iTriangle < startIndex + count; ++iTriangle)
            {
                Triangle* triangle = &bvh->m_Triangles[bvh->m_TriangleIdx[iTriangle]];
                if (triangle->m_Pos.y > splitPoint)
                {
                    // Calculate new AABB
                    float leftVolume, rightVolume;
                    int leftCount = (iTriangle - startIndex);
                    if (leftCount == 0)
                    {
                        leftVolume = 0;
                    }
                    else
                    {
                        AABB left = AABB(bvh->m_Triangles, bvh->m_TriangleIdx, startIndex, leftCount);
                        leftVolume = left.CalculateVolume();
                    }

                    int rightCount = count - leftCount;
                    if (rightCount == 0)
                    {
                        rightVolume = 0;
                    }
                    else
                    {

                        AABB right = AABB(bvh->m_Triangles, bvh->m_TriangleIdx, iTriangle, rightCount);
                        rightVolume = right.CalculateVolume();
                    }
                    float cost = leftVolume * leftCount + rightVolume * rightCount;
                    if (cost < bestCost)
                    {
                        bestCost = cost;
                        // New start index
                        bestSplitIndex = iTriangle;
                        bestAxis = 1;
                    }
                    break;
                }
            }
            splitPoint += posStep;
        }
        splitIndexY = bestSplitIndex;
    }
    //else
    {
        // Split on z
        float posStep = dz / 10.0f;
        SortZ(bvh->m_Triangles, bvh->m_TriangleIdx, firstLeft, firstLeft + count - 1);
        // Find best splitpoint
        // Offset now
        float splitPoint = posStep + m_AABB.m_Min.z;
        int startIndex = firstLeft;
        int bestSplitIndex = startIndex;
        for (int i = 0; i < 8; ++i)
        {
            // Find point where we surpass the split point
            for (int iTriangle = startIndex; iTriangle < startIndex + count; ++iTriangle)
            {
                Triangle* triangle = &bvh->m_Triangles[bvh->m_TriangleIdx[iTriangle]];
                if (triangle->m_Pos.z > splitPoint)
                {
                    // Calculate new AABB
                    float leftVolume, rightVolume;
                    int leftCount = (iTriangle - startIndex);
                    if (leftCount == 0)
                    {
                        leftVolume = 0;
                    }
                    else
                    {
                        AABB left = AABB(bvh->m_Triangles, bvh->m_TriangleIdx, startIndex, leftCount);
                        leftVolume = left.CalculateVolume();
                    }

                    int rightCount = count - leftCount;
                    if (rightCount == 0)
                    {
                        rightVolume = 0;
                    }
                    else
                    {

                        AABB right = AABB(bvh->m_Triangles, bvh->m_TriangleIdx, iTriangle, rightCount);
                        rightVolume = right.CalculateVolume();
                    }
                    float cost = leftVolume * leftCount + rightVolume * rightCount;
                    if (cost < bestCost)
                    {
                        bestCost = cost;
                        // New start index
                        bestSplitIndex = iTriangle;
                        bestAxis = 2;
                    }
                    break;
                }
            }
            splitPoint += posStep;
        }
        splitIndexZ = bestSplitIndex;
    }
    int splitIndex;
    if (bestAxis == 0)
    {
        SortX(bvh->m_Triangles, bvh->m_TriangleIdx, firstLeft, firstLeft + count - 1);
        splitIndex = splitIndexX;
    }
    else if (bestAxis == 1)
    {
        SortY(bvh->m_Triangles, bvh->m_TriangleIdx, firstLeft, firstLeft + count - 1);
        splitIndex = splitIndexY;
    }
    else
    {
        splitIndex = splitIndexZ;
    }
    // Z is already sorted
    if (initCost == bestCost)// Don't split
    {
        return;
    }
    leftNode->firstLeft = firstLeft;
    leftNode->count = splitIndex - firstLeft;
    rightNode->firstLeft = splitIndex;
    rightNode->count = count - (splitIndex - firstLeft);
    assert(leftNode->count != 0 && rightNode->count != 0);/*
    
    if (leftNode->count == 0 || rightNode->count == 0)
    {
        return;
    }*/

    firstLeft = bvh->poolPtr - 1; // Save our left node index
    count = 0; // We're not a leaf
    leftNode->Subdivide(bvh, depth + 1);
    rightNode->Subdivide(bvh, depth + 1);
}
BOOL BVHNode::Traverse(BVH* bvh, Ray & a_Ray, BVHResult& a_Result)
{
    if (!CheckBox(m_AABB.m_Min, m_AABB.m_Max, a_Ray.O, a_Ray.D, a_Ray.t))
        return FALSE;
	if (count != 0)// Leaf
	{
        return IntersectPrimitives(bvh, a_Ray, a_Result);
	}

    // small optimization -> check which one is closer first, because it has a higher chance that the ray'll hit an object closer to the ray origin.
	return bvh->m_Nodes[firstLeft + 1].Traverse(bvh, a_Ray, a_Result) | bvh->m_Nodes[firstLeft].Traverse(bvh, a_Ray, a_Result);
}
BOOL BVHNode::TraverseDepth(BVH * bvh, Ray & a_Ray, int & depth, BVHResult& a_Result)
{
	depth++;
	// Maybe invert?? No raymi u drunk.
    if (!CheckBox(m_AABB.m_Min, m_AABB.m_Max, a_Ray.O, a_Ray.D, a_Ray.t))
        return FALSE;
	if (count != 0)// Leaf
	{
        return IntersectPrimitives(bvh, a_Ray, a_Result);
	}
	return bvh->m_Nodes[firstLeft + 1].TraverseDepth(bvh, a_Ray, depth, a_Result) | bvh->m_Nodes[firstLeft].TraverseDepth(bvh, a_Ray, depth, a_Result);
}
BOOL BVHNode::IntersectPrimitives(BVH * bvh, Ray & a_Ray, BVHResult& a_Result)
{
	// Loop through the triangles
    Triangle* triangle;
	float u, v;
	for (int i = firstLeft; i < firstLeft + count; ++i)
	{
        triangle = &bvh->m_Triangles[bvh->m_TriangleIdx[i]];
        if (triangle->Intersect(a_Ray, u, v))
        {
            a_Result.m_Triangle = triangle;
            a_Result.m_U = u;
            a_Result.m_V = v;
            
            return TRUE;
        }
	}
	return FALSE;
}
