#pragma once

//*******************************************************************

#define MAXOBJECTS 1024
#define PACKETSIZE 64

//*******************************************************************

namespace Tmpl8
{

    //
    // Primitive types
    //
    class Object;
    class Sphere;
    class Plane;
    class MeshCollider;
    class Triangle;

    //
    // Ray tracing
    //
    class Ray;
    class Raytracer;

    //
    // BVH
    //
    struct BVHResult;
    class  BVHNode;
    class  BVH;
    class  AABB;


    typedef vec4 Color;
	struct Color64
	{
		Color color[PACKETSIZE];
	};
	struct BOOL64
	{
		BOOL bools[PACKETSIZE];
	};
    typedef struct
    {
        vec3  m_Pos;
        Color m_Color;
    } Light;

    //*******************************************************************

    //========================
    //         OBJECT
    //========================
    class Object
    {

    public:
        Object(const vec3& a_Pos, const Color& a_Diffuse, float a_Refl, float a_Refr, float a_RIndex)
            : m_Pos(a_Pos)
            , m_Diffuse(a_Diffuse)
            , m_Refl(a_Refl)
            , m_Refr(a_Refr)
            , m_RIndex(a_RIndex)
        {}

        virtual vec3 HitNormal(const vec3& a_Pos) { return vec3(0, 0, 0); }
        virtual BOOL Intersect(Ray& a_Ray)        { return FALSE;         }
        virtual BOOL IsOccluded(const Ray& a_Ray) { return FALSE;         }

    public:
        vec3  m_Pos;      // World-space position
        Color m_Diffuse;  // Color and transparency
        float m_Refl;     // Reflection
        float m_Refr;     // Refraction
        float m_RIndex;   // Refraction index
    };

    //========================
    //         SPHERE
    //========================
    class Sphere : Object
    {

    public:
        Sphere(vec3& a_Pos, vec4& a_Diffuse, float a_Refl, float a_Refr, float a_RIndex, float a_Radius)
            : Object(a_Pos, a_Diffuse, a_Refl, a_Refr, a_RIndex)
            , m_Radius(a_Radius)
            , m_SqRadius(a_Radius*a_Radius)
        {}

       virtual vec3 HitNormal(const vec3& a_Pos);
       virtual BOOL Intersect(Ray& a_Ray);
       virtual BOOL IsOccluded(const Ray& a_Ray);

    public:
        float m_Radius;
        float m_SqRadius;

    };

    //========================
    //         PLANE
    //========================
    class Plane : Object
    {

    public:
        Plane(const vec3& a_Normal, const vec4& a_Color, float a_Refl, float a_Refr, float a_RIndex, float a_Dist)
            : Object(a_Normal, a_Color, a_Refl, a_Refr, a_RIndex)
            , m_Dist(a_Dist)
        {}

        vec3 HitNormal(const vec3& a_Pos);
        BOOL Intersect(Ray& a_Ray);
        BOOL IsOccluded(const Ray& a_Ray);

    public:
        float m_Dist;

    };

    //========================
    //     MESHCOLLIDER
    //========================
    class MeshCollider : public Object
    {
    
    public:
        MeshCollider(const vec3& a_Pos, const vec4& a_Color, float a_Refl, float a_Refr, float a_RIndex, Mesh* a_Mesh)
            : Object(a_Pos, a_Color, a_Refl, a_Refr, a_RIndex)
            , m_Mesh(a_Mesh)
        {}

        vec3 HitNormal(const vec3& a_Pos);
        BOOL Intersect(Ray& a_Ray);
        BOOL IsOccluded(const Ray& a_Ray);

    public:
        Mesh*    m_Mesh;
        Texture* m_Tex;
        int      index;
    
    };

    //========================
    //        TRIANGLE
    //========================
    class Triangle : public Object
    {
    public:
        Triangle(const vec3& a_Pos, const vec4& a_Color, float a_Refl, float a_Refr, float a_RIndex, Mesh* a_Mesh, int idx)
            : Object(a_Pos, a_Color, a_Refl, a_Refr, a_RIndex)
        {
            // get all data
            normal = a_Mesh->N[idx];
            p0 = a_Mesh->pos[a_Mesh->tri[idx * 3]];
            p1 = a_Mesh->pos[a_Mesh->tri[idx * 3 + 1]];
            p2 = a_Mesh->pos[a_Mesh->tri[idx * 3 + 2]];

            // centre point
            m_Pos = (p0 + p1 + p2) / 3.0f;

            uv0 = a_Mesh->uv[a_Mesh->tri[idx * 3]];
            uv1 = a_Mesh->uv[a_Mesh->tri[idx * 3 + 1]];
            uv2 = a_Mesh->uv[a_Mesh->tri[idx * 3 + 2]];
            m_Tex = a_Mesh->material->texture;
            index = idx;

        }

        vec3 HitNormal(const vec3& a_Pos);
        BOOL Intersect(Ray& a_Ray, float& a_U, float& a_V);
		BOOL Intersect2(vec3& D, vec3& O, float& t, float& a_U, float& a_V);
        BOOL IsOccluded(const Ray& a_Ray);

        vec4 GetColor(float a_U, float a_V);


    public:
        vec3 normal;
        vec3 p0, p1, p2;
        vec2 uv0, uv1, uv2;
        int index;
        Texture* m_Tex;
        // Texture* n_Normals

    };

    //========================
    //          RAY
    //========================
    class Ray
    {

    public:
        // constructor / destructor
        Ray( vec3 origin, vec3 direction, float distance ) : O( origin ), D( direction ), t( distance ) {}
        // data members
        vec3 O, D;
        float t;

    };

    struct RayPacket
    {
        vec3 O[PACKETSIZE], D[PACKETSIZE], rD[PACKETSIZE];
        float t[PACKETSIZE];
        //int firstActive;
    };

    //========================
    //       RAYTRACER
    //========================
    class Raytracer
    {
    public:
        // constructor / destructor
        Raytracer() : scene( 0 ), traverseDepth(0){}
        ~Raytracer() { _aligned_free(frameBuffer); }
        // methods
        void  Init( Surface* screen );
        BOOL  IsOccluded( Ray& ray );
        void  Render( Camera& camera );
        void  RenderScanlines(Camera& camera);
        Color64 TraceRayPacket(RayPacket& rayPacket, int firstActive);
        Color GetColorFromSphere(Ray& a_Ray, int& a_ReflPass, int& a_RefrPass, float a_RIndex);
        vec4  GetBVHDepth(Ray& ray,int& depth);
        void  BuildBVH(vector<Mesh*> meshList);

        // data members
        Scene*   scene;	
        Surface* screen;
        vec3     screenCenter;
        Pixel*   frameBuffer;
        BVH*     bvh;
        int      curLine;
        bool	 traverseDepth;
        
        // Scene objects
        vector<Object*> m_Objects;
        vector<Light*>  m_Lights;
    };

    //========================
    //         AABB
    //========================

    class AABB
    {
    public:
        vec3 m_Min = vec3(INFINITY);
        vec3 m_Max = vec3(-INFINITY);
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
            assert(count > 0);
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
        vec3 GetDelta()
        {
            return m_Max - m_Min;
        }
        float CalculateVolume()
        {
            vec3 delta = GetDelta();
            return delta.x * delta.y * delta.z;
        }
        float CalculateSurfaceArea()
        {
            vec3 delta = GetDelta();
            return (delta.z*delta.x + delta.z*delta.y + delta.x* delta.y) * 2;
        }
    };

    //========================
    //        BVHRESULT
    //========================
    struct BVHResult
    {
        Triangle* m_Triangle;
        float   m_U, m_V;
    };
    struct BVHResultPacket
    {
        Triangle* m_Triangle[PACKETSIZE];
        float   m_U[PACKETSIZE], m_V[PACKETSIZE];
    };

    //========================
    //         BVHNODE
    //========================
    class BVHNode
    {
    public:
        AABB m_AABB;
        glm::uint32_t firstLeft;
        glm::uint32_t count;

        void Subdivide(BVH* bvh, int depth);
        BOOL Traverse(BVH* bvh, Ray& a_Ray, BVHResult& a_Result);
        BOOL TraverseDepth(BVH* bvh, Ray& a_Ray, int& depth, BVHResult& a_Result);
		void TraversePacket(BVH* bvh, RayPacket& rayPacket, BVHResultPacket& resultPacket, int firstActive);
        BOOL IntersectPrimitives(BVH* bvh, Ray& a_Ray, BVHResult& a_Result);
		void IntersectPrimitivesPacket(BVH* bvh, RayPacket& rayPacket, BVHResultPacket& resultPacket, int firstActive);

    };

    //========================
    //           BVH
    //========================
    class BVH
    {
    public:
        BVH(vector<Mesh*> meshes);
        BOOL Traverse(Ray& a_Ray, BVHResult& a_Result);
        BOOL TraverseDepth(Ray& a_Ray, int& depth, BVHResult& a_Result);
		void TraverseRayPacket(RayPacket& rayPacket, BVHResultPacket& resultPacket, int firstActive);
        const unsigned int primPerNode = 3;
        unsigned int nNodes;// amount of nodes
        unsigned int nTris;
        unsigned int poolPtr;
        unsigned int* m_TriangleIdx;
        Triangle* m_Triangles;
        BVHNode* m_Nodes;

        int maxDepth;
    };

}; // namespace Tmpl8
