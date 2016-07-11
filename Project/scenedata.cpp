#include "template.h"


// -----------------------------------------------------------
// Tiny functions
// -----------------------------------------------------------
Texture::Texture(char* file) : name(0) { pixels = new Surface8(file); SetName(file); }
Texture::~Texture() { delete pixels; delete name; }
void Texture::SetName(const char* n) { delete name; strcpy(name = new char[strlen(n) + 1], n); }
Material::~Material() { delete name; }
void Material::SetName(char* n) { delete name; strcpy(name = new char[strlen(n) + 1], n); }
SGNode::~SGNode() { for (uint i = 0; i < child.size(); i++) delete child[i]; }
Mesh::~Mesh() { delete pos; delete N; delete spos; delete tri; }

// -----------------------------------------------------------
// static data
// -----------------------------------------------------------
Surface* Mesh::screen = 0;
float* Mesh::xleft, *Mesh::xright, *Mesh::uleft, *Mesh::uright;
float* Mesh::vleft, *Mesh::vright, *Mesh::zleft, *Mesh::zright;
static vec3 raxis[3] = { vec3(1, 0, 0), vec3(0, 1, 0), vec3(0, 0, 1) };

// -----------------------------------------------------------
// Scene destructor
// -----------------------------------------------------------
Scene::~Scene()
{
    delete root;
    for (uint i = 0; i < texList.size(); i++) delete texList[i];
    for (uint i = 0; i < matList.size(); i++) delete matList[i];
    delete scenePath;
}

// -----------------------------------------------------------
// Scene::ExtractPath
// retrieves the path from a file name;
// used to construct the path to mesh textures
// -----------------------------------------------------------
void Scene::ExtractPath(const char* file)
{
    char tmp[2048], *lastSlash = tmp;
    strcpy(tmp, file);
    while (strstr(lastSlash, "/")) lastSlash = strstr(lastSlash, "/") + 1;
    while (strstr(lastSlash, "\\")) lastSlash = strstr(lastSlash, "\\") + 1;
    *lastSlash = 0;
    delete scenePath;
    scenePath = new char[strlen(tmp) + 1];
    strcpy(scenePath, tmp);
}

// -----------------------------------------------------------
// Scene::FindMaterial
// get a material pointer by material name
// -----------------------------------------------------------
Material* Scene::FindMaterial(const char* name)
{
    for (uint s = matList.size(), i = 0; i < s; i++) if (matList[i]->name)
        if (!strcmp(matList[i]->name, name)) return matList[i];
    return 0;
}

// -----------------------------------------------------------
// Scene::FindTexture
// get a texture pointer by texture name
// -----------------------------------------------------------
Texture* Scene::FindTexture(const char* name)
{
    for (uint s = texList.size(), i = 0; i < s; i++) if (texList[i]->name)
        if (!strcmp(texList[i]->name, name)) return texList[i];
    return 0;
}

// -----------------------------------------------------------
// Scene::LoadMTL
// loads a MTL file for an OBJ file
// -----------------------------------------------------------
void Scene::LoadMTL(const char* file)
{
    char fname[1024], line[1024], cmd[256], matName[128];
    strcpy(fname, scenePath);
    if (fname[0]) strcat(fname, "/");
    strcat(fname, file);
    FILE* f = fopen(fname, "r");
    if (!f) return;
    Material* current = 0;
    int firstIdx = matList.size();
    while (!feof(f))
    {
        line[0] = 0;
        fgets(line, 1022, f);
        if (!line[0]) continue;
        if (line[strlen(line) - 1] < 32) line[strlen(line) - 1] = 0; // clean '10' at end
        sscanf(line, "%s", cmd);
        if (!_stricmp(cmd, "newmtl"))
        {
            sscanf(line + strlen(cmd), "%s", matName);
            matList.push_back(current = new Material());
            current->SetName(matName);
        }
        if (_stricmp(cmd, "map_Kd")) continue;
        char* tname = strstr(line, " ");
        if (!tname) continue; else tname++;
        strcpy(fname, scenePath);
        strcat(fname, "textures/");
        strcat(fname, tname);
        Texture* texture = FindTexture(fname);
        if (!texture) texList.push_back(texture = new Texture(fname));
        current->texture = texture;
    }
    fclose(f);
}

// -----------------------------------------------------------
// Scene::LoadOBJ
// loads an OBJ file, returns a scene graph node
// a scene graph node is produced for each mesh in the obj
// file, and for each unique material in each mesh. the
// resulting scene graph is ready for rendering in a state-
// driven renderer (e.g. OGL).
// -----------------------------------------------------------
SGNode* Scene::LoadOBJ(const char* file, const float _Scale)
{
    struct UniqueVertex
    {
        UniqueVertex(int v, int n = -1, int t = -1) : vertex(v), normal(n), uv(t), next(-1), subid(0) {}
        int vertex, normal, uv, next, subid, idx;
    };
    SGNode* root = new SGNode(), *group = root;
    Mesh* current = 0, *nextMesh = 0;
    FILE* f = fopen(file, "r");
    ExtractPath(file);
    if (f) // obj file loader: converts indexed obj file into indexed multi-mesh
    {
        vector<vec3> vlist, nlist, vlist_, nlist_;
        vector<vec2> uvlist, uvlist_;
        vector<uint> index, index_;
        vector<UniqueVertex> unique;
        vlist.reserve(100000);
        nlist.reserve(100000);
        unique.reserve(100000);
        int subID = 1, formata;
        while (!feof(f))
        {
            current = nextMesh, subID++, formata = -1;
            bool hasUV = false;
            char line[2048], tmp[2048];
            while (!feof(f))
            {
                line[0] = 0;
                fgets(line, 1023, f);
                if (!_strnicmp(line, "mtllib", 6))
                {
                    sscanf(line + 7, "%s", tmp);
                    LoadMTL(tmp);
                    formata = -1, hasUV = false;
                }
                if (!_strnicmp(line, "usemtl", 6))
                {
                    // prepare new mesh
                    sscanf(line + 7, "%s", tmp);
                    nextMesh = new Mesh();
                    nextMesh->material = FindMaterial(tmp);
                    group->child.push_back(nextMesh);
                    subID++;
                    break;
                }
                if (line[0] == 'g')
                {
                    formata = -1;
                    char* g = line + 2;
                    while ((g[0]) && (g[strlen(g) - 1] < 33)) g[strlen(g) - 1] = 0;
                    root->child.push_back(group = new SGNode());
                }
                if (line[0] == 'v') if (line[1] == ' ')
                {
                    vec3 vertex;
                    sscanf(line + 2, "%f %f %f", &vertex.x, &vertex.y, &vertex.z);
                    vlist.push_back(vertex * _Scale);
                    unique.push_back(UniqueVertex(vlist.size() - 1));
                }
                else if (line[1] == 't') { vec2 uv; sscanf(line + 3, "%f %f", &uv.x, &uv.y); uv.y = 1.0f - uv.y; uvlist.push_back(uv); }
                else if (line[1] == 'n') { vec3 normal; sscanf(line + 3, "%f %f %f", &normal.x, &normal.y, &normal.z); nlist.push_back(normal); }
                if (line[0] != 'f') continue;
                if (formata == -1)
                {
                    formata = 0;
                    for (int i = 0; i < (int)strlen(line); i++) if (line[i] == '/' && line[i + 1] == '/') formata = 1;
                }
                int v[3], n[3], t[3] = { 0, 0, 0 };
                if (formata) sscanf(line + 2, "%i//%i %i//%i %i//%i", &v[0], &n[0], &v[1], &n[1], &v[2], &n[2]);
                sscanf(line + 2, "%i/%i/%i %i/%i/%i %i/%i/%i", &v[0], &t[0], &n[0], &v[1], &t[1], &n[1], &v[2], &t[2], &n[2]);
                for (int i = 0; i < 3; i++)
                {
                    int vidx = v[i] - 1, idx = vidx, lastIdx = idx, nidx = n[i] - 1, uvidx = t[i] - 1;
                    if (uvidx > -1) hasUV = true;
                    do
                    {
                        UniqueVertex& u = unique[idx];
                        if (u.subid != subID) // vertex not used before by this mesh
                        {
                            u.subid = subID, u.next = -1, u.vertex = vidx, u.normal = nidx, u.uv = uvidx;
                            index.push_back(idx);
                            break;
                        }
                        else if ((u.normal == nidx) && (u.uv == uvidx)) // vertex used before, but the same
                        {
                            index.push_back(idx);
                            break;
                        }
                        lastIdx = idx, idx = u.next;
                    } while (idx > -1);
                    if (idx != -1) continue;
                    uint newIdx = unique.size();
                    unique[lastIdx].next = newIdx;
                    index.push_back(newIdx);
                    unique.push_back(UniqueVertex(vidx, nidx, uvidx));
                    unique[newIdx].subid = subID;
                }
            }

            if (!current) continue; else subID++;
            vlist_.clear();
            nlist_.clear();
            uvlist_.clear();
            index_.clear();
            for (uint i = 0; i < index.size(); i++)
            {
                UniqueVertex& u = unique[index[i]];
                if (u.subid == subID) index_.push_back(u.idx); else // first time we encounter this UniqueVertex, emit
                {
                    vlist_.push_back(vlist[u.vertex]);
                    nlist_.push_back(nlist[u.normal]);
                    if (hasUV) uvlist_.push_back(uvlist[u.uv]);
                    else uvlist_.push_back(vec2(0, 1));
                    index_.push_back(vlist_.size() - 1);
                    u.idx = vlist_.size() - 1, u.subid = subID;
                }
            }
            // create mesh
            int nv = current->verts = vlist_.size(), nt = current->tris = index_.size() / 3;
            current->pos = new vec3[nv * 3], current->tpos = current->pos + nv;
            current->norm = current->pos + 2 * nv, current->spos = new vec2[nv * 2];
            current->N = new vec3[nt], current->uv = current->spos + nv;
            current->tri = new int[nt * 3];
            memcpy(current->pos, (vec3*)&vlist_[0], current->verts * sizeof(vec3));
            memcpy(current->uv, (vec2*)&uvlist_[0], current->verts * sizeof(vec2));
            memcpy(current->tri, (int*)&index_[0], current->tris * 3 * sizeof(int));
            memcpy(current->norm, (vec3*)&nlist_[0], current->verts * sizeof(vec3));
            meshList.push_back(current);
            // calculate triangle planes
            for (int i = 0; i < current->tris; i++)
            {
                vec3 v0 = vlist_[index_[i * 3 + 0]], v1 = vlist_[index_[i * 3 + 1]], v2 = vlist_[index_[i * 3 + 2]];
                current->N[i] = normalize(cross(v1 - v0, v2 - v0));
                if (dot(current->N[i], nlist_[index_[i * 3 + 1]]) < 0) current->N[i] *= -1.0f;
            }
            // calculate mesh bounds
            vec3& bmin = current->bounds[0], &bmax = current->bounds[1];
            bmin = { 1e30f, 1e30f, 1e30f }, bmax = { -1e30f, -1e30f, -1e30f };
            for (int i = 0; i < current->verts; i++)
                bmin.x = min(bmin.x, current->pos[i].x), bmax.x = max(bmax.x, current->pos[i].x),
                bmin.y = min(bmin.y, current->pos[i].y), bmax.y = max(bmax.y, current->pos[i].y),
                bmin.z = min(bmin.z, current->pos[i].z), bmax.z = max(bmax.z, current->pos[i].z);
            // clean up
            while (unique.size() > vlist.size()) unique.pop_back();
            index.clear();
        }
        fclose(f);
    }
    return root;
}

// -----------------------------------------------------------
// Mesh constructor
// input: vertex count & face count
// allocates room for mesh data: 
// - pos:  vertex positions
// - tpos: transformed vertex positions
// - norm: vertex normals
// - spos: vertex screen space positions
// - uv:   vertex uv coordinates
// - N:    face normals
// - tri:  connectivity data
// -----------------------------------------------------------
Mesh::Mesh(int vcount, int tcount) : verts(vcount), tris(tcount)
{
    pos = new vec3[vcount * 3], tpos = pos + vcount, norm = pos + 2 * vcount;
    spos = new vec2[vcount * 2], uv = spos + vcount, N = new vec3[tcount];
    tri = new int[tcount * 3];
}

// -----------------------------------------------------------
// SGNode::RotateABC
// helper function for the RotateXYZ permutations
// -----------------------------------------------------------
void SGNode::RotateABC(float a, float b, float c, int a1, int a2, int a3)
{
    mat4 M = rotate(rotate(rotate(mat4(), a, raxis[a1]), b, raxis[a2]), c, raxis[a3]);
    for (int x = 0; x < 3; x++) for (int y = 0; y < 3; y++) localTransform[x][y] = M[x][y];
}