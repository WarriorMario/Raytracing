// Template, major revision 6
// IGAD/NHTV - Jacco Bikker - 2006-2015

#pragma once

#define MAXJOBTHREADS	32
#define MAXJOBS			512

class Thread 
{
public:
	Thread() { m_hThread = 0; }
	unsigned long* handle() { return m_hThread; }
	void start();
	virtual void run() {};
	void sleep(long ms);
	void suspend();
	void resume();
	void kill();
	void stop();
	void setPriority( int p );
	void SetName( char* _Name );
private:
	unsigned long* m_hThread;
	static const int P_ABOVE_NORMAL;
	static const int P_BELOW_NORMAL;
	static const int P_HIGHEST;
	static const int P_IDLE;
	static const int P_LOWEST;
	static const int P_NORMAL;
	static const int P_CRITICAL;
};
extern "C" { unsigned int sthread_proc( void* param ); }

namespace Tmpl8 {
	class BVH;
	class Camera;
	class Raytracer;
class Job
{
public:
	virtual void Main() = 0;
protected:
	friend class JobThread;
	void RunCodeWrapper();
};

class JobThread
{
public:
	void CreateAndStartThread( unsigned int threadId );
	void WaitForThreadToStop();
	void Go();
	void BackgroundTask();
	HANDLE m_GoSignal, m_ThreadHandle;
	int m_ThreadID;
};

class JobManager	// singleton class!
{
protected:
	JobManager( unsigned int numThreads );
public:
	~JobManager();
	static void CreateJobManager( unsigned int numThreads );
	static JobManager* GetJobManager() { return m_JobManager; }
	void AddJob2( Job* a_Job );
	unsigned int GetNumThreads() { return m_NumThreads; }
	void RunJobs();
	void ThreadDone( unsigned int n );
	int MaxConcurrent() { return m_NumThreads; }
	int numCores;
protected:
	friend class JobThread;
	Job* GetNextJob();
	Job* FindNextJob();
	static JobManager* m_JobManager;
	Job* m_JobList[MAXJOBS];
	CRITICAL_SECTION m_CS;
	HANDLE m_ThreadDone[MAXJOBTHREADS];
	unsigned int m_NumThreads, m_JobCount;
	JobThread* m_JobThreadList;
};

struct RenderData
{
	glm::vec3 p0;
	glm::vec3 p1;
	glm::vec3 p2;

	float invHeight;
	float invWidth;
};

class RenderJob : public Job
{
public:
	static volatile long waiting;
	static glm::ivec2 tiles[SCRWIDTH * SCRHEIGHT];
	static RenderData renderData;
	void Main();
	void Init(BVH * a_bvh, Camera * camera, Surface * a_screen, Raytracer* rayt);
	void RenderTile(int tile);

protected:
 	BVH * bvh;
	Camera * camera;
	Surface * screen;
	Raytracer * raytracer;
};

}; // namespace Tmpl8