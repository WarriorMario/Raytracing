#pragma once

namespace Tmpl8
{

    #define MAXOBJECTS 1024
    #define NUMLIGHTS 1

    
    class Object;
    class Sphere;
    class Plane;
    class Ray;
    class Raytracer;
    class BVHNode;
    class BVH;
    class AABB;

    class Object
    {

    public:
        Object(vec3& a_Pos, vec4& a_Diffuse, float a_Refl, float a_Refr)
            : m_Pos(a_Pos)
            , m_Diffuse(a_Diffuse)
            , m_Refl(a_Refl)
            , m_Refr(a_Refr)
        {}

        virtual vec3 HitNormal(const vec3& a_Pos) { return vec3(0, 0, 0); }
        virtual BOOL Intersect(Ray& a_Ray)        { return FALSE;         }
        virtual BOOL IsOccluded(const Ray& a_Ray) { return FALSE;         }
        virtual vec4 GetColor()                   { return m_Diffuse;     }

    public:
        vec3  m_Pos;
        vec4  m_Diffuse;
        float m_Refl;
        float m_Refr;
    };

    class Sphere : Object
    {

    public:
        Sphere(vec3& a_Pos, vec4& a_Diffuse, float a_Refl, float a_Refr, float a_Radius)
            : Object(a_Pos, a_Diffuse, a_Refl, a_Refr)
            , m_Radius(a_Radius)
        {}

        virtual vec3 HitNormal(const vec3& a_Pos);
        virtual BOOL Intersect(Ray& a_Ray);
        virtual BOOL IsOccluded(const Ray& a_Ray);
        virtual vec4 GetColor();

    public:
        float m_Radius;

    };
    class Plane : Object
    {

    public:
        Plane(vec3& a_Normal, vec4& a_Diffuse, float a_Refl, float a_Refr, float a_Dist)
            : Object(a_Normal, a_Diffuse, a_Refl, a_Refr)
            , m_Dist(a_Dist)
        {}

        virtual vec3 HitNormal(const vec3& a_Pos);
        virtual BOOL Intersect(Ray& a_Ray);
        virtual BOOL IsOccluded(const Ray& a_Ray);
        virtual vec4 GetColor();

    public:
        float m_Dist;

    };
    
    class MeshCollider : public Object
    {
    public:
        MeshCollider(vec3& a_Normal, vec4& a_Diffuse, float a_Refl, float a_Refr, Mesh* a_Mesh)
            : Object(a_Normal, a_Diffuse, a_Refl, a_Refr)
            , m_Mesh(a_Mesh)
        {}

        virtual vec3 HitNormal(const vec3& a_Pos);
        virtual BOOL Intersect(Ray& a_Ray);
        virtual BOOL IsOccluded(const Ray& a_Ray);
        virtual vec4 GetColor();

        Mesh* m_Mesh;
        Texture* m_Tex;
        int index;
    };

    class Triangle : public Object
    {
    public:
        Triangle(vec3& a_Normal, vec4& a_Diffuse, float a_Refl, float a_Refr, Mesh* a_Mesh, int idx)
            : Object(a_Normal, a_Diffuse, a_Refl, a_Refr)
        {
            // get all data
            normal = a_Mesh->norm[idx];
        }

        virtual vec3 HitNormal(const vec3& a_Pos);
        virtual BOOL Intersect(Ray& a_Ray);
        BOOL Intersect(Ray& a_Ray, float& u, float&v);
        virtual BOOL IsOccluded(const Ray& a_Ray);
        virtual vec4 GetColor();
        void Init(Mesh* mesh, int idx)
        {
            normal = mesh->N[idx];
            p0 = mesh->pos[mesh->tri[idx * 3]];
            p1 = mesh->pos[mesh->tri[idx * 3 + 1]];
            p2 = mesh->pos[mesh->tri[idx * 3 + 2]];
            // centre point
            m_Pos = (p0 + p1 + p2) / 3.0f;

            uv0 = mesh->uv[mesh->tri[idx * 3]];
            uv1 = mesh->uv[mesh->tri[idx * 3 + 1]];
            uv2 = mesh->uv[mesh->tri[idx * 3 + 2]];
            m_Tex = mesh->material->texture;
			this->mesh = mesh;
			triID = idx;
        }
		int triID;
        vec3 normal;
        vec3 p0, p1, p2;
        vec2 uv0, uv1, uv2;
		float u, v;
		Mesh* mesh;
        Texture* m_Tex;
        // Texture* n_Normals
    };

    // -----------------------------------------------------------
    // Raytracer struct
    // generic ray
    // -----------------------------------------------------------
    class Ray
    {
    public:
        // constructor / destructor
        Ray( vec3 origin, vec3 direction, float distance ) : O( origin ), D( direction ), t( distance ) {}
        // data members
        vec3 O, D;
        float t;
    };

    // -----------------------------------------------------------
    // Raytracer class
    // to be build
    // -----------------------------------------------------------
    class Raytracer
    {
    public:
        // constructor / destructor
        Raytracer() : scene( 0 ) {}
        ~Raytracer() { _aligned_free(frameBuffer); }
        // methods
        void Init( Surface* screen );
        void FindNearest( Ray& ray );
        BOOL IsOccluded( Ray& ray );
        void Render( Camera& camera );
        void RenderScanlines(Camera& camera);
        vec4 CheckHit(Ray& ray, int exception, int recursion);
        vec4 GetColorFromSphere(Ray& ray, int exception);
		vec4 GetBVHDepth(Ray& ray,int& depth);
        void BuildBVH(vector<Mesh*> meshList);

        // data members
        Scene* scene;	
        Surface* screen;
        vec3 screenCenter;
        Pixel* frameBuffer;
        BVH* bvh;
        int curLine;
        
        // Scene objects
        vector<Object*> objects;
        vec3 lights[NUMLIGHTS];
        vec4 lightColors[NUMLIGHTS];
    };

    class AABB
    {
    public:
        vec3 m_Min;
        vec3 m_Max;
        AABB(const Triangle& t)
        {
            // Get the bounds
            m_Min.x = min(t.m_Pos.x, min(t.p1.x, t.p2.x));
            m_Max.x = max(t.m_Pos.x, max(t.p1.x, t.p2.x));
            m_Min.y = min(t.m_Pos.y, min(t.p1.y, t.p2.y));
            m_Max.y = max(t.m_Pos.y, max(t.p1.y, t.p2.y));
            m_Min.z = min(t.m_Pos.z, min(t.p1.z, t.p2.z));
            m_Max.z = max(t.m_Pos.z, max(t.p1.z, t.p2.z));
        }
        AABB(const AABB* aabbs, int count)
        {
            for (int i = 0; i < count; ++i)
            {
                m_Min.x = min(aabbs[i].m_Min.x, m_Min.x);
                m_Min.y = min(aabbs[i].m_Min.y, m_Min.y);
                m_Min.z = min(aabbs[i].m_Min.z, m_Min.z);
                m_Max.x = max(aabbs[i].m_Max.x, m_Max.x);
                m_Max.y = max(aabbs[i].m_Max.y, m_Max.y);
                m_Max.z = max(aabbs[i].m_Max.z, m_Max.z);
            }
        }
        AABB(Triangle* triangles, unsigned int* indexArray, unsigned int start, unsigned int count)
        {
            for (unsigned int i = start; i < start + count; ++i)
            {
                unsigned int index = indexArray[i];
                Triangle& t = triangles[index];

                m_Min.x = min(min(t.p0.x, min(t.p1.x, t.p2.x)), m_Min.x);
                m_Min.y = min(min(t.p0.y, min(t.p1.y, t.p2.y)), m_Min.y);
                m_Min.z = min(min(t.p0.z, min(t.p1.z, t.p2.z)), m_Min.z);
                m_Max.x = max(max(t.p0.x, max(t.p1.x, t.p2.x)), m_Max.x);
                m_Max.y = max(max(t.p0.y, max(t.p1.y, t.p2.y)), m_Max.y);
                m_Max.z = max(max(t.p0.z, max(t.p1.z, t.p2.z)), m_Max.z);
            }
        }
		float CalculateVolume()
		{
			vec3 delta = m_Max - m_Min;
			return delta.x * delta.x + delta.y *delta.y + delta.z*delta.z;
		}
    };

    class BVHNode
    {
    public:
        AABB m_AABB;
        int firstLeft;
        int count;

        void Subdivide(BVH* bvh);
        vec4 Traverse(BVH* bvh, Ray& a_Ray);
		vec4 TraverseDepth(BVH* bvh, Ray& a_Ray, int& depth);
        vec4 IntersectPrimitives(BVH* bvh, Ray& a_Ray);

    };

    class BVH
    {
    public:
        BVH(vector<Mesh*> meshes);
        vec4 Traverse(Ray& a_Ray);
		vec4 TraverseDepth(Ray& a_Ray, int& depth);
        const unsigned int primPerNode = 30;
        unsigned int nNodes;// amount of nodes
        unsigned int nTris;
        unsigned int poolPtr;
        unsigned int* m_TriangleIdx;
        Triangle* m_Triangles;
        BVHNode* m_Nodes;
    };
}; // namespace Tmpl8
