#pragma once


struct ProfileScope
{
    ProfileScope(const char* name);
    ~ProfileScope();
};

#define PROFILE_SCOPE_VAR(line) _profiler_##line
#define PROFILE_SCOPE(name) ProfileScope PROFILE_SCOPE_VAR(__LINE__)(name)

namespace profiler 
{
    void register_current_thread(const char* name);
}
