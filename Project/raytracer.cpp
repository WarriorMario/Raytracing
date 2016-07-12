#include "template.h"

//*************************************************

#define BACKGROUND vec4(1.0f, 0.3f, 0.3f, 0.3f)
#define AMBIENT vec4(1.0f,0.5f,0.5f,0.5f)
#define TEST 1
constexpr float kEpsilon = 1e-8;
//*************************************************

using namespace Tmpl8; 
int biggest = 0;

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

vec3 Refract(vec3 N, vec3 I, float n1, float n2)
{
	// Credits to crazy maths guy.
	float ndoti, two_ndoti, ndoti2, a, b, b2, D2;
	vec3 T;
	ndoti = N.x*I.x + N.y*I.y + N.z*I.z;     // 3 mul, 2 add
	ndoti2 = ndoti*ndoti;                    // 1 mul
	if (ndoti >= 0.0)
	{
		b = n1 / n2;
		b2 = b2 * b2;
	}
	else
	{
		b = n2 / n1;
		b2 = b2 * b2;
	}
	D2 = 1.0f - b2*(1.0f - ndoti2);

	if (D2 >= 0.0f) {
		if (ndoti >= 0.0f)
			a = b * ndoti - sqrtf(D2); // 2 mul, 3 add, 1 sqrt
		else
			a = b * ndoti + sqrtf(D2);
		T.x = a*N.x - b*I.x;     // 6 mul, 3 add
		T.y = a*N.y - b*I.y;     // ----totals---------
		T.z = a*N.z - b*I.z;     // 12 mul, 8 add, 1 sqrt!
	}
	else {
		// total internal reflection
		// this usually doesn't happen, so I don't count it.
		two_ndoti = ndoti + ndoti;         // +1 add
		T.x = two_ndoti * N.x - I.x;      // +3 adds, +3 muls
		T.y = two_ndoti * N.y - I.y;
		T.z = two_ndoti * N.z - I.z;
	}
	return T;
}

// ============================================
// Object occlusion test
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

vec4 Tmpl8::Sphere::GetColor()
{
	return m_Diffuse;
}

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
	if (t < 0.01) return FALSE;
}

vec4 Tmpl8::Plane::GetColor()
{
	return m_Diffuse;
}


// ============================================
// Ray tracer manager
// ============================================
void Raytracer::Init(Surface * screen)
{
	this->screen = screen;
	this->frameBuffer = (Pixel*)_aligned_malloc(screen->GetHeight() * screen->GetWidth() * sizeof(Pixel), 32);
	memset(frameBuffer, 0, screen->GetHeight() * screen->GetWidth() * sizeof(Pixel));
	this->curLine = 0;

	for (int i = 0; i < NUMLIGHTS; i++)
	{
		lightColors[i] = vec4(1.0f, 1.0f, 1.0f, 1.0f);
		lights[i] = vec3(0, 5, 0);
	}

	const int numSpheres = 2;
	const int numPlanes = 1;

	for (int i = 0; i < numSpheres; i++)
	{
		Sphere* sphere = new Sphere(vec3(i, i, i), PixelToVec(0xff00ff00), 0.4f, 0.0f, (float)i / (float)numSpheres);
		objects.push_back((Object*)sphere);
	}
	objects[1]->m_Refl = 0.0f;
	objects[1]->m_Refr = 0.0f;
	//for (int i = 0; i < numPlanes; i++)
	//{
	//    Plane* plane = new Plane(vec3(0, 1, 0), PixelToVec(0xff00ff00), 0.2f, 0.0f, true, false);
	//    objects.push_back((Object*)plane);
	//}
}

void Raytracer::FindNearest(Ray & ray)
{
}

BOOL Raytracer::IsOccluded(Ray & ray)
{
	for (int i = 0; i < objects.size(); i++)
	{
		if (objects[i]->IsOccluded(ray)) return TRUE;
	}

	return FALSE;
}

int reflPasses = 0;
int refrPasses = 0;
float maxT = -200000.0f;
float minT = 0.0f;
void Raytracer::Render(Camera & camera)
{
	float fov = 1.8f;
	//screenCenter = camera.GetForward() * fov;// +camera.GetPosition();
	vec3 p0 = vec3(camera.transform * vec4(vec3(-1, SCRASPECT, -fov), 1));// -camera.GetPosition());
	vec3 p1 = vec3(camera.transform * vec4(vec3(1, SCRASPECT, -fov), 1));// -camera.GetPosition());
	vec3 p2 = vec3(camera.transform * vec4(vec3(-1, -SCRASPECT, -fov), 1));// -camera.GetPosition());
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
			Ray ray = Ray(camera.GetPosition(), normalize(planepos - camera.GetPosition()), 100000000);

			reflPasses = 0;
			refrPasses = 0;
			//screen->GetBuffer()[x + line] = VecToPixel(GetColorFromSphere(ray, -1));
			int depth = 0;
			vec4  color = GetBVHDepth(ray,depth);
			screen->GetBuffer()[x + line] = VecToPixel(vec4(1, (float)depth / 60, 1.0f - (float)depth / 60, 0));
			maxT = max((float)depth, maxT);
		}
	}
}

void Tmpl8::Raytracer::RenderScanlines(Camera & camera)
{
	float fov = 1.8f;
	//screenCenter = camera.GetForward() * fov;// +camera.GetPosition();
	vec3 p0 = vec3(camera.transform * vec4(vec3(-1, SCRASPECT, -fov), 1));// -camera.GetPosition());
	vec3 p1 = vec3(camera.transform * vec4(vec3(1, SCRASPECT, -fov), 1));// -camera.GetPosition());
	vec3 p2 = vec3(camera.transform * vec4(vec3(-1, -SCRASPECT, -fov), 1));// -camera.GetPosition());
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

		reflPasses = 0;
		refrPasses = 0;
		int depth = 0;
		//vec4  color = GetBVHDepth(ray,depth);
		//frameBuffer[x + line] = VecToPixel(vec4(1, (float)depth / 20, 1.0f - (float)depth / 20, 0));
		frameBuffer[x + line] = VecToPixel(GetColorFromSphere(ray, -1));
		if (depth > 1)
		{
			int i = 0;
		}
		// Draw black sphere around light.
		vec3 c = lights[0] - ray.O;
		float t = dot(c, ray.D);
		if (t < 0)continue;
		vec3 q = c - t*ray.D;
		float p2 = dot(q, q);
		if (p2 < 1) screen->GetBuffer()[x + line] = 0;
	}
	memcpy(screen->GetBuffer(), frameBuffer, screen->GetHeight() * screen->GetWidth() * sizeof(Pixel));
	if (curLine < screen->GetHeight() - 1)
		curLine++;
}

vec4 Raytracer::GetColorFromSphere(Ray & ray, int exception)
{
	return bvh->Traverse(ray);


	int closestIndex = -1;
	for (int i = 0; i < objects.size(); i++)
	{
		if (i == exception) continue;

		if (objects[i]->Intersect(ray))
		{
			closestIndex = i;
		}
	}
	if (closestIndex < 0)
	{
		return BACKGROUND;
	}


	vec3 hit = ray.O + ray.D * ray.t;
	vec3 normal = objects[closestIndex]->HitNormal(hit);
	vec4 finalColor = vec4(0.0f);
	vec4 colorP = vec4(1, 1, 1, 1);// objects[closestIndex]->m_Diffuse;//*diffuse;

	// Reflect.
	if (objects[closestIndex]->m_Refl > 0.0f)
	{
		Ray reflRay = Ray(hit, Reflect(ray.D, normal), INFINITY);
		vec4 reflColor;
		if (reflPasses < 5)
		{
			reflPasses++;
			reflColor = GetColorFromSphere(reflRay, closestIndex);
		}
		finalColor += reflColor * colorP * objects[closestIndex]->m_Refl;
	}

	if (objects[closestIndex]->m_Refr > 0.0f)
	{
		// Refract.
		const float n1 = 1.0f, n2 = 0.5f;
		Ray refrRay = Ray(hit, Refract(normal, ray.D, n1, n2), INFINITY);
		vec4 refrColor;
		if (refrPasses < 5)
		{
			refrPasses++;
			refrColor = GetColorFromSphere(refrRay, -1);
		}
		finalColor += refrColor * colorP * objects[closestIndex]->m_Refr;
	}

	float lightStrength = 1.0f - (objects[closestIndex]->m_Refr + objects[closestIndex]->m_Refl);
	if (lightStrength > 0.0f)
	{
		finalColor += AMBIENT * colorP * lightStrength;
		for (int i = 0; i < NUMLIGHTS; ++i)
		{
			Ray lightRay = Ray(hit, normalize(lights[i] - hit), INFINITY);
			if (IsOccluded(lightRay) == false)
			{
				float d = dot(normal, lightRay.D);
				if (d < 0.0f)
				{
					d = 0.0f;
				}
				finalColor += d*lightColors[i] * colorP * lightStrength;
			}
		}
	}

	//float reflectance = objects[closestIndex]->m_Refl;
	//float refractance = 0.4f;
	//float diffuse = 1.0f - reflectance;// -refractance;
	//vec4 colorR = reflColor * reflectance;
	//vec4 colorT = refrColor * refractance;

	//vec4 finalColor = lightColor;// +colorR*colorP;
	//vec4 finalColor = (AMBIENT + lightColor) * (colorR + colorP);

	return clamp(finalColor, 0.0f, 1.0f);
}

vec4 Tmpl8::Raytracer::GetBVHDepth(Ray & ray, int& depth)
{
	return bvh->TraverseDepth(ray, depth);
}

void Tmpl8::Raytracer::BuildBVH(vector<Mesh*> meshList)
{
	bvh = new BVH(meshList);
}

bool rayTriangleIntersect(Ray & ray,
	const vec3  &v0, const vec3 &v1, const vec3 &v2,
	float &u, float &v)
{
	vec3 v0v1 = v1 - v0;
	vec3 v0v2 = v2 - v0;
	vec3 pvec = cross(ray.D, v0v2);
	float det = dot(v0v1, pvec);

	// if the determinant is negative the triangle is backfacing
	// if the determinant is close to 0, the ray misses the triangle
	if (det < -kEpsilon) return FALSE;
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

bool rayTriangleHit(const Ray & ray, const vec3 &v0, const vec3 &v1, const vec3 &v2)
{
	vec3 v0v1 = v1 - v0;
	vec3 v0v2 = v2 - v0;
	vec3 pvec = cross(ray.D, v0v2);
	float det = dot(v0v1, pvec);

	// if the determinant is negative the triangle is backfacing
	// if the determinant is close to 0, the ray misses the triangle
	if (det < -kEpsilon) return FALSE;
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

vec3 Tmpl8::MeshCollider::HitNormal(const vec3 & a_Pos)
{
	return m_Mesh->N[index];
}

BOOL Tmpl8::MeshCollider::IsOccluded(const Ray & a_Ray)
{
	bool hit = FALSE;
	for (int i = 0; i < m_Mesh->tris; i++)
	{
		// Get the triangle
		vec3 p0 = m_Mesh->pos[m_Mesh->tri[i * 3]];
		vec3 p1 = m_Mesh->pos[m_Mesh->tri[i * 3 + 1]];
		vec3 p2 = m_Mesh->pos[m_Mesh->tri[i * 3 + 2]];
		if (rayTriangleHit(a_Ray, p0, p1, p2))
		{
			return TRUE;
		}
	}
	return FALSE;
}

vec4 Tmpl8::MeshCollider::GetColor()
{
	return vec4();
}

BOOL Tmpl8::MeshCollider::Intersect(Ray& a_Ray)
{
	bool hit = FALSE;
	for (int i = 0; i < m_Mesh->tris; i++)
	{
		// Get the triangle
		vec3 p0 = m_Mesh->pos[m_Mesh->tri[i * 3]];
		vec3 p1 = m_Mesh->pos[m_Mesh->tri[i * 3 + 1]];
		vec3 p2 = m_Mesh->pos[m_Mesh->tri[i * 3 + 2]];
		float u, v;
		if (rayTriangleIntersect(a_Ray, p0, p1, p2, u, v))
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

vec3 Tmpl8::Triangle::HitNormal(const vec3 & a_Pos)
{
	return vec3();
}

BOOL Tmpl8::Triangle::Intersect(Ray & a_Ray)
{
	return 0;
}
// Return should be a struct containing u v etc
BOOL Tmpl8::Triangle::Intersect(Ray & a_Ray, float & a_U, float & a_V)
{
	vec3 v0v1 = p1 - p0;
	vec3 v0v2 = p2 - p0;
	vec3 pvec = cross(a_Ray.D, v0v2);
	float det = dot(v0v1, pvec);
	float u, v;
	// if the determinant is negative the triangle is backfacing
	// if the determinant is close to 0, the ray misses the triangle
	if (det < -kEpsilon) return FALSE;
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

BOOL Tmpl8::Triangle::IsOccluded(const Ray & a_Ray)
{
	return 0;
}

vec4 Tmpl8::Triangle::GetColor()
{
	return vec4();
}

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
float triangleCenterX[5000];
float triangleSortedX[5000];
Tmpl8::BVH::BVH(vector<Mesh*> meshes)
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
		for (int j = 0; j < meshes[i]->tris; ++j)
		{
			m_Triangles[trIdx].Init(meshes[i], j);
			m_TriangleIdx[trIdx] = trIdx;
			triangleCenterX[trIdx] = m_Triangles[trIdx].m_Pos.x;
			trIdx++;
		}
	}

	// check the size of BVHNode
	size_t size = sizeof(BVHNode);
	nNodes = nTris * 2 + 1;
	m_Nodes = (BVHNode*)_aligned_malloc(nNodes * sizeof(BVHNode), 128);

	BVHNode* root = &m_Nodes[0];
	poolPtr = 0;
	root->firstLeft = 0;
	root->count = nTris;
	root->Subdivide(this,0); // construct the first node
	int test = 0;

}
bool CheckBox(vec3& bmin, vec3& bmax, vec3 O, vec3 rD, float t)
{
	vec3 tMin = (bmin - O) / rD, tMax = (bmax - O) / rD;
	vec3 t1 = min(tMin, tMax), t2 = max(tMin, tMax);
	float tNear = max(max(t1.x, t1.y), t1.z);
	float tFar = min(min(t2.x, t2.y), t2.z);
	return ((tFar > tNear) && (tNear < t) && (tFar > 0));
}


BoxCheck GetFromBox(AABB& box, vec3 O, vec3 rD, float t)
{
	BoxCheck ret;

		vec3 tMin = (box.m_Min - O) / rD, tMax = (box.m_Max - O) / rD;
		vec3 t1 = min(tMin, tMax), t2 = max(tMin, tMax);
		ret.tNear = max(max(t1.x, t1.y), t1.z);
		ret.tFar = min(min(t2.x, t2.y), t2.z);
		ret.hit = ((ret.tFar > ret.tNear) && (ret.tNear < t) && (ret.tFar > 0));
		return ret;
}
vec4 Tmpl8::BVH::Traverse(Ray & a_Ray)
{
	return m_Nodes[0].Traverse(this, a_Ray);
}

vec4 Tmpl8::BVH::TraverseDepth(Ray & a_Ray, int& depth)
{
#if TEST
	return m_Nodes[0].TraverseDepth(this, a_Ray, depth);
#else
	BoxCheck check = GetFromBox(m_Nodes[0].m_AABB, a_Ray.O, a_Ray.D, a_Ray.t);
	return m_Nodes[0].TraverseDepth(this, a_Ray, depth,check);
#endif
}

void Tmpl8::BVHNode::Subdivide(BVH * bvh,int depth)
{
	maxT = max(maxT, (float)depth);
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
		if (leftNode->count > biggest)
		{
			biggest = leftNode->count;
		}
		if (rightNode->count > biggest)
		{
			biggest = rightNode->count;
		}
		return;
	}
	leftNode->firstLeft = firstLeft;
	leftNode->count = splitIndex - firstLeft;
	rightNode->firstLeft = splitIndex;
	rightNode->count = count - (splitIndex - firstLeft);
	if (leftNode->count == 0 || rightNode->count == 0)
	{
		if (leftNode->count > biggest)
		{
			biggest = leftNode->count;
		}
		if (rightNode->count > biggest)
		{
			biggest = rightNode->count;
		}
		return;
	}
	firstLeft = bvh->poolPtr - 1; // Save our left node index
	count = 0; // We're not a leaf
	leftNode->Subdivide(bvh, depth + 1);
	if (leftNode->count > biggest)
	{
		biggest = leftNode->count;
	}
	rightNode->Subdivide(bvh, depth + 1); 
	if (rightNode->count > biggest)
	{
		biggest = rightNode->count;
	}
}

vec4 Tmpl8::BVHNode::Traverse(BVH* bvh, Ray & a_Ray)
{
	if (!CheckBox(m_AABB.m_Min, m_AABB.m_Max, a_Ray.O, a_Ray.D, a_Ray.t))
		return BACKGROUND;
	if (count != 0)// Leaf
	{
		return IntersectPrimitives(bvh, a_Ray);
	}
	vec4 color = bvh->m_Nodes[firstLeft].Traverse(bvh, a_Ray);
	vec4 color2 = bvh->m_Nodes[firstLeft + 1].Traverse(bvh, a_Ray);
	if (color != BACKGROUND)
	{
		return color;
	}
	return color2;
}
vec4 Tmpl8::BVHNode::TraverseDepth(BVH * bvh, Ray & a_Ray, int & depth)
{
	if (bvh->test)
	{
		// Maybe invert?? 
		if (!CheckBox(m_AABB.m_Min, m_AABB.m_Max, a_Ray.O, a_Ray.D, a_Ray.t))
			return BACKGROUND;
		depth++;
		if (count != 0)// Leaf
		{
			return IntersectPrimitives(bvh, a_Ray);
		}
		vec4 color = bvh->m_Nodes[firstLeft].TraverseDepth(bvh, a_Ray, depth);
		vec4 color2 = bvh->m_Nodes[firstLeft + 1].TraverseDepth(bvh, a_Ray, depth);

		if (color == BACKGROUND)
		{
			return color2;
		}
		return color;
	}
	else
	{
		// Maybe invert?? 
		if (!CheckBox(m_AABB.m_Min, m_AABB.m_Max, a_Ray.O, a_Ray.D, a_Ray.t))
			return BACKGROUND;
		depth++;
		if (count != 0)// Leaf
		{
			return IntersectPrimitives(bvh, a_Ray);
		}
		vec3 delta1 = bvh->m_Nodes[firstLeft].m_AABB.m_Max - bvh->m_Nodes[firstLeft+1].m_AABB.m_Min;
		vec3 delta2 = bvh->m_Nodes[firstLeft+1].m_AABB.m_Max - bvh->m_Nodes[firstLeft].m_AABB.m_Min;


		vec4 color = bvh->m_Nodes[firstLeft].TraverseDepth(bvh, a_Ray, depth);
		vec4 color2 = bvh->m_Nodes[firstLeft + 1].TraverseDepth(bvh, a_Ray, depth);

		if (color == BACKGROUND)
		{
			return color2;
		}
		return color;
	}
}

vec4 Tmpl8::BVHNode::IntersectPrimitives(BVH * bvh, Ray & a_Ray)
{
	// Loop through the triangles
	Triangle* test;
	float u, v;
	bool hit = false;
	for (int i = firstLeft; i < firstLeft + count; ++i)
	{
		test = &bvh->m_Triangles[bvh->m_TriangleIdx[i]];
		if (test->Intersect(a_Ray, u, v))
		{
			hit = true;
		}
	}
	if (hit)
	{
		return vec4(1, 1, 1, 1);
	}
	return BACKGROUND;
}
