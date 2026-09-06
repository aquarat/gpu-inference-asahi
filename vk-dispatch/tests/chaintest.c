/*
 * Per-dispatch cost of a chain of N compute dispatches, in two shapes:
 *   dependent   : every dispatch reads+writes the SAME buffer region, with a
 *                 vkCmdPipelineBarrier (COMPUTE shader write -> COMPUTE shader
 *                 read, buffer memory barrier) between consecutive dispatches.
 *                 This is the shape of an inference graph (layer i+1 consumes
 *                 layer i). Result is verified: each element must equal N
 *                 (with loops=0) after the chain.
 *   independent : each dispatch touches its own slice, no barrier at all.
 * Usage: ./chaintest dep|indep N groups loops [reps]
 * Prints GPU time (timestamps) and CPU wall (submit->idle) per dispatch in us.
 * Build: glslangValidator -V chaintest.comp -o chaintest.spv
 *        cc -O2 -o chaintest chaintest.c -lvulkan
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>
#define CHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
    fprintf(stderr, "%s:%d: %s -> %d\n", __FILE__, __LINE__, #x, _r); exit(1);} } while (0)
static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
static uint32_t *load_spv(const char *path, size_t *len){ FILE *f=fopen(path,"rb"); if(!f){perror(path);exit(1);} fseek(f,0,SEEK_END); *len=ftell(f); fseek(f,0,SEEK_SET); uint32_t *b=malloc(*len); if(fread(b,1,*len,f)!=*len){perror("read");exit(1);} fclose(f); return b; }
int main(int argc, char **argv){
   if (argc < 5) { fprintf(stderr, "usage: %s dep|indep N groups loops [reps]\n", argv[0]); return 2; }
   int dep = strcmp(argv[1], "dep") == 0; unsigned N = atoi(argv[2]), groups = atoi(argv[3]), loops = atoi(argv[4]);
   int reps = argc > 5 ? atoi(argv[5]) : 3;
   const unsigned LS = 64; unsigned per = groups * LS;
   VkInstance inst; VkApplicationInfo app={.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,.apiVersion=VK_API_VERSION_1_3};
   CHECK(vkCreateInstance(&(VkInstanceCreateInfo){.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,.pApplicationInfo=&app},NULL,&inst));
   uint32_t n=0; vkEnumeratePhysicalDevices(inst,&n,NULL); VkPhysicalDevice *pds=calloc(n,sizeof(*pds)); vkEnumeratePhysicalDevices(inst,&n,pds);
   VkPhysicalDevice pd=pds[0]; for(uint32_t i=0;i<n;i++){VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(pds[i],&p); if(p.deviceType!=VK_PHYSICAL_DEVICE_TYPE_CPU){pd=pds[i];break;}}
   VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(pd,&props);
   VkPhysicalDeviceDriverProperties drv={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
   VkPhysicalDeviceProperties2 p2={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,.pNext=&drv}; vkGetPhysicalDeviceProperties2(pd,&p2);
   printf("device: %s  driver: %s\n", props.deviceName, drv.driverInfo);
   float prio=1; VkDevice dev;
   VkPhysicalDeviceSynchronization2Features s2={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,.synchronization2=VK_TRUE};
   CHECK(vkCreateDevice(pd,&(VkDeviceCreateInfo){.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,.pNext=&s2,.queueCreateInfoCount=1,.pQueueCreateInfos=&(VkDeviceQueueCreateInfo){.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,.queueFamilyIndex=0,.queueCount=1,.pQueuePriorities=&prio}},NULL,&dev));
   VkQueue q; vkGetDeviceQueue(dev,0,0,&q);
   VkDeviceSize size = (VkDeviceSize)(dep ? per : per * N) * 4; if (size < 4096) size = 4096;
   VkBuffer buf; CHECK(vkCreateBuffer(dev,&(VkBufferCreateInfo){.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,.size=size,.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT},NULL,&buf));
   VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev,buf,&mr); VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd,&mp);
   uint32_t mt=UINT32_MAX; for(uint32_t i=0;i<mp.memoryTypeCount;i++) if((mr.memoryTypeBits&(1u<<i)) && (mp.memoryTypes[i].propertyFlags&(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))==(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)){mt=i;break;}
   if(mt==UINT32_MAX){fprintf(stderr,"no host-visible coherent memory\n");return 1;}
   VkDeviceMemory mem; CHECK(vkAllocateMemory(dev,&(VkMemoryAllocateInfo){.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,.allocationSize=mr.size,.memoryTypeIndex=mt},NULL,&mem));
   CHECK(vkBindBufferMemory(dev,buf,mem,0)); uint32_t *map; CHECK(vkMapMemory(dev,mem,0,size,0,(void**)&map));
   VkDescriptorSetLayoutBinding b={.binding=0,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=1,.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT};
   VkDescriptorSetLayout dsl; CHECK(vkCreateDescriptorSetLayout(dev,&(VkDescriptorSetLayoutCreateInfo){.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,.bindingCount=1,.pBindings=&b},NULL,&dsl));
   VkPushConstantRange pcr={.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,.offset=0,.size=8};
   VkPipelineLayout pl; CHECK(vkCreatePipelineLayout(dev,&(VkPipelineLayoutCreateInfo){.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,.setLayoutCount=1,.pSetLayouts=&dsl,.pushConstantRangeCount=1,.pPushConstantRanges=&pcr},NULL,&pl));
   size_t spvlen; uint32_t *code=load_spv("chaintest.spv",&spvlen); VkShaderModule sm;
   CHECK(vkCreateShaderModule(dev,&(VkShaderModuleCreateInfo){.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,.codeSize=spvlen,.pCode=code},NULL,&sm));
   VkPipeline pipe; CHECK(vkCreateComputePipelines(dev,VK_NULL_HANDLE,1,&(VkComputePipelineCreateInfo){.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,.stage={.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_COMPUTE_BIT,.module=sm,.pName="main"},.layout=pl},NULL,&pipe));
   VkDescriptorPoolSize ps={.type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=1}; VkDescriptorPool dp;
   CHECK(vkCreateDescriptorPool(dev,&(VkDescriptorPoolCreateInfo){.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,.maxSets=1,.poolSizeCount=1,.pPoolSizes=&ps},NULL,&dp));
   VkDescriptorSet ds; CHECK(vkAllocateDescriptorSets(dev,&(VkDescriptorSetAllocateInfo){.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,.descriptorPool=dp,.descriptorSetCount=1,.pSetLayouts=&dsl},&ds));
   vkUpdateDescriptorSets(dev,1,&(VkWriteDescriptorSet){.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=ds,.dstBinding=0,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&(VkDescriptorBufferInfo){.buffer=buf,.offset=0,.range=VK_WHOLE_SIZE}},0,NULL);
   VkQueryPool qp; CHECK(vkCreateQueryPool(dev,&(VkQueryPoolCreateInfo){.sType=VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,.queryType=VK_QUERY_TYPE_TIMESTAMP,.queryCount=2},NULL,&qp));
   VkCommandPool cp; CHECK(vkCreateCommandPool(dev,&(VkCommandPoolCreateInfo){.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,.queueFamilyIndex=0},NULL,&cp));
   VkCommandBuffer cb; CHECK(vkAllocateCommandBuffers(dev,&(VkCommandBufferAllocateInfo){.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,.commandPool=cp,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1},&cb));
   VkBufferMemoryBarrier bmb={.sType=VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT,.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.buffer=buf,.offset=0,.size=VK_WHOLE_SIZE};
   printf("%-6s %6s %7s %6s %10s %10s %12s %12s %8s\n","mode","N","groups","loops","gpu_ms","cpu_ms","gpu_us/disp","cpu_us/disp","verify");
   for (int rep = 0; rep < reps; rep++) {
      memset(map, 0, size);
      CHECK(vkResetCommandPool(dev,cp,0));
      double t0 = now_us();
      CHECK(vkBeginCommandBuffer(cb,&(VkCommandBufferBeginInfo){.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT}));
      vkCmdResetQueryPool(cb,qp,0,2);
      vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pipe);
      vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pl,0,1,&ds,0,NULL);
      vkCmdWriteTimestamp(cb,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,qp,0);
      for (unsigned d = 0; d < N; d++) {
         struct { uint32_t off, loops; } pc = { dep ? 0 : d * per, loops };
         vkCmdPushConstants(cb,pl,VK_SHADER_STAGE_COMPUTE_BIT,0,8,&pc);
         vkCmdDispatch(cb,groups,1,1);
         if (dep && d + 1 < N)
            vkCmdPipelineBarrier(cb,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,0,NULL,1,&bmb,0,NULL);
      }
      vkCmdWriteTimestamp(cb,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,qp,1);
      CHECK(vkEndCommandBuffer(cb));
      double t1 = now_us();
      CHECK(vkQueueSubmit(q,1,&(VkSubmitInfo){.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&cb},VK_NULL_HANDLE));
      CHECK(vkQueueWaitIdle(q));
      double t2 = now_us();
      uint64_t ts[2]; CHECK(vkGetQueryPoolResults(dev,qp,0,2,sizeof(ts),ts,8,VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WAIT_BIT));
      double gpu_ms = (ts[1]-ts[0]) * props.limits.timestampPeriod / 1e6;
      /* verify (loops==0 only): dependent -> every element == N ; independent -> every element == 1 */
      const char *ver = "n/a"; if (loops == 0) { unsigned bad = 0; unsigned expect = dep ? N : 1; size_t cnt = dep ? per : (size_t)per * N; for (size_t i = 0; i < cnt; i++) if (map[i] != expect) bad++; ver = bad ? "FAIL" : "ok"; if (bad) printf("  VERIFY FAIL: %u of %zu elements wrong (map[0]=%u expect %u)\n", bad, cnt, map[0], expect); }
      printf("%-6s %6u %7u %6u %10.3f %10.3f %12.2f %12.2f %8s   (record %.1f ms)\n", dep?"dep":"indep", N, groups, loops, gpu_ms, (t2-t1)/1e3, gpu_ms*1e3/N, (t2-t1)/N, ver, (t1-t0)/1e3);
   }
   return 0;
}
