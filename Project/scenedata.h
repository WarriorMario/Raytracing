#pragma once


namespace Tmpl8
{
    // -----------------------------------------------------------
    // The beautifull contents of this file.
    // -----------------------------------------------------------
    class Material;
    class Texture;
    class Scene;
    class Camera;
    class SGNode;
    class Mesh;

    // -----------------------------------------------------------
    // Material class
    // basic material properties
    // -----------------------------------------------------------
    class Material
    {
    public:
        // constructor / destructor
        Material() : texture(0), name(0) {}
        ~Material();
        // methods
        void SetName(char* name);
        // data members
        uint diffuse;					// diffuse material color
        Texture* texture;				// texture
        char* name;						// material name
    };

    // -----------------------------------------------------------
    // Texture class
    // encapsulates a palettized pixel surface with pre-scaled
    // palettes for fast shading
    // -----------------------------------------------------------
    class Texture
    {
    public:
        // constructor / destructor
        Texture() : name(0) {}
        Texture(char* file);
        ~Texture();
        // methods
        void SetName(const char* name);
        void Load(const char* file);
        // data members
        char* name;
        Surface8* pixels;
    };

    // -----------------------------------------------------------
    // SGNode class
    // scene graph node, with convenience functions for translate
    // and transform; base class for Mesh
    // -----------------------------------------------------------
    class SGNode
    {
    public:
        enum
        {
            SG_TRANSFORM = 0,
            SG_MESH
        };
        // constructor / destructor
        ~SGNode();
        // methods
        void SetPosition(vec3& pos) { mat4& M = localTransform; M[3][0] = pos.x, M[3][1] = pos.y, M[3][2] = pos.z; }
        vec3 GetPosition() { mat4& M = localTransform; return vec3(M[3][0], M[3][1], M[3][2]); }
        void RotateX(float x) { RotateABC(x, 0, 0, 0, 1, 2); }
        void RotateY(float y) { RotateABC(y, 0, 0, 1, 0, 2); }
        void RotateZ(float z) { RotateABC(z, 0, 0, 2, 1, 0); }
        void RotateXYZ(float x, float y, float z) { RotateABC(x, y, z, 0, 1, 2); }
        void RotateXZY(float x, float y, float z) { RotateABC(x, z, y, 0, 2, 1); }
        void RotateYXZ(float x, float y, float z) { RotateABC(y, x, z, 1, 0, 2); }
        void RotateZXY(float x, float y, float z) { RotateABC(z, x, y, 2, 0, 1); }
        void RotateYZX(float x, float y, float z) { RotateABC(y, z, x, 1, 2, 0); }
        void RotateZYX(float x, float y, float z) { RotateABC(z, y, x, 2, 1, 0); }
        void Add(SGNode* node) { child.push_back(node); }
        virtual int GetType() { return SG_TRANSFORM; }
    private:
        void RotateABC(float a, float b, float c, int a1, int a2, int a3);
        // data members
    public:
        mat4 localTransform;
        vector<SGNode*> child;
    };

    // -----------------------------------------------------------
    // Scene class
    // owner of the scene graph;
    // owner of the material and texture list
    // -----------------------------------------------------------
    class Scene
    {
    public:
        // constructor / destructor
        Scene() : root(0), scenePath(0) {}
        ~Scene();
        // methods
        SGNode* Add(char* file, float scale = 1.0f) { SGNode* n = LoadOBJ(file, scale); root->Add(n); return n; }
        SGNode* LoadOBJ(const char* file, const float scale);
        Material* FindMaterial(const char* name);
        Texture* FindTexture(const char* name);
    private:
        void ExtractPath(const char* file);
        void LoadMTL(const char* file);
        // data members
    public:
        SGNode* root;
        vector<Mesh*> meshList;
        vector<Material*> matList;
        vector<Texture*> texList;
        char* scenePath;
    };

    // -----------------------------------------------------------
    // Camera class
    // convenience class for storing the camera transform
    // -----------------------------------------------------------
    class Camera
    {
    public:
        // methods
        void SetPosition(vec3& pos) { mat4& M = transform; M[3][0] = pos.x, M[3][1] = pos.y, M[3][2] = pos.z; }
        vec3 GetPosition() { mat4& M = transform; return vec3(M[3][0], M[3][1], M[3][2]); }
        vec3 GetRight() { mat4& M = transform; return vec3(M[0][0], M[0][1], M[0][2]); }
        vec3 GetUp() { mat4& M = transform; return vec3(M[1][0], M[1][1], M[1][2]); }
        vec3 GetForward() { mat4& M = transform; return -vec3(M[2][0], M[2][1], M[2][2]); }
        void LookAt(vec3 target) { transform = inverse(lookAt(GetPosition(), target, vec3(0, 1, 0))); }
        // data members
        mat4 transform;
    };

    // -----------------------------------------------------------
    // Mesh class
    // represents a mesh
    // -----------------------------------------------------------
    class Mesh : public SGNode
    {
    public:
        // constructor / destructor
        Mesh() : verts(0), tris(0), pos(0), uv(0), spos(0) {}
        Mesh(int vcount, int tcount);
        ~Mesh();
        virtual int GetType() { return SG_MESH; }
        // data members
        vec3* pos;						// object-space vertex positions
        vec3* tpos;						// world-space positions
        vec2* uv;						// vertex uv coordinates
        vec2* spos;						// screen positions
        vec3* norm;						// vertex normals
        vec3* N;						// triangle plane
        int* tri;						// connectivity data
        int verts, tris;				// vertex & triangle count
        Material* material;				// mesh material
        vec3 bounds[2];					// mesh bounds
        static Surface* screen;
        static float* xleft, *xright;	// outline tables for rasterization
        static float* uleft, *uright;
        static float* vleft, *vright;
        static float* zleft, *zright;
    };

    //class Sphere : public SGNode
    //{
    //    // constructor / destructor
    //    Sphere() : pos(0) {}
    //    ~Sphere();
    //    virtual int GetType() { return SG_MESH; }
    //    // data members
    //    vec3 pos;						// world-space position
    //    int norm;						// sphere normal
    //    Material* material;				// mesh material
    //    vec3 bounds[2];					// mesh bounds
    //    static Surface* screen;
    //    static float* xleft, *xright;	// outline tables for rasterization
    //    static float* uleft, *uright;
    //    static float* vleft, *vright;
    //    static float* zleft, *zright;
    //};

}