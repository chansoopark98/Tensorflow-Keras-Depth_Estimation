import open3d as o3d
import copy 
from modern_robotics import *

# original
# voxel_size = 0.001
# icp_distance = 0.01
# color_icp_distance = 0.02

# test
voxel_size = 0.001
icp_distance = voxel_size * 100
color_icp_distance = voxel_size * 50

def cal_angle(pl_norm, R_dir):
    angle_in_radians = \
        np.arccos(
            np.abs(pl_norm.x*R_dir[0]+ pl_norm.y*R_dir[1] + pl_norm.z*R_dir[2])
            )

    return angle_in_radians

def preprocess_point_cloud(pcd, voxel_size):
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    # pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down = pcd

    radius_normal = 0.1
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 15
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)

    """Original RANSAC"""
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    
    """Test RANSAC"""
    # result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    #     source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint(
    #         False), 4,
    #     [o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
    #         distance_threshold)],
    #     o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.8))
    
    return result

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      )
    
def registerLocalCloud(target, source):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)

        source_temp, source_fpfh = preprocess_point_cloud(source_temp, voxel_size)
        target_temp, target_fpfh = preprocess_point_cloud(target_temp, voxel_size)

        
        result_ransac = execute_global_registration(source_temp, target_temp,
                                                    source_fpfh, target_fpfh,
                                                    voxel_size)
        print("RANSAC Result")
        # draw_registration_result(source_temp, target_temp, result_ransac.transformation)
        
        # current_transformation = np.identity(4)
        current_transformation = result_ransac.transformation
        
        result_icp_p2l = o3d.pipelines.registration.registration_icp(source_temp, target_temp, icp_distance,
                current_transformation, o3d.pipelines.registration.TransformationEstimationPointToPlane())


        print("Plane ICP Result")
        # draw_registration_result(source_temp, target_temp, result_icp_p2l.transformation)

        p2l_init_trans_guess = result_icp_p2l.transformation

        result_icp = o3d.pipelines.registration.registration_icp(source_temp, target_temp, color_icp_distance,
                p2l_init_trans_guess,  o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                )

        print("Color ICP Result")
        # draw_registration_result(source_temp, target_temp, result_icp.transformation)

        tf = result_icp.transformation
        R = tf[:3,:3]  # rotation matrix
        so3mat = MatrixLog3(R)
        omg = so3ToVec(so3mat)
        
        trans_tol= 0.5  # transformation tolerance
        
        print('tf', tf)
        print('icp fitness {0}'.format(result_icp.fitness))
        if ( tf[0,3] > trans_tol or tf[0,3] < -trans_tol or \
             tf[1,3] > trans_tol or tf[1,3] < -trans_tol or \
             tf[2,3] > trans_tol or tf[2,3] < -trans_tol ):
            print('bad result')
            print('icp fitness {0}'.format(result_icp.fitness))
            return np.identity(4)

        return result_icp.transformation


if __name__ == '__main__':

    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    # Load point clouds
    pcds = []
    dir_name = '2023_03_20_16_48_56'
    for i in range(23):
        pcd = o3d.io.read_point_cloud('./360degree_pointclouds/{1}/pcd/test_pointcloud_{0}.pcd'.format(i, dir_name))
        print(np.mean(np.asarray(pcd.points)[:, 2]))

        # o3d.visualization.draw_geometries([pcd])
        
        # Filtering
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
        # cl, ind = pcd.remove_radius_outlier(nb_points=16, radius=0.2)
        filteredpcd = pcd.select_by_index(ind)
        filteredpcd = filteredpcd.voxel_down_sample(voxel_size)

        

        # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, -0.01])
        # o3d.visualization.draw_geometries([filteredpcd])

        pcds.append(filteredpcd)
    

    # Visualize the mesh
    o3d.visualization.draw_geometries(pcds)

    cloud_base = pcds[0]

    cloud1 = copy.deepcopy(cloud_base)


    detectTransLoop = np.identity(4)
    posWorldTrans = np.identity(4)

    for cloud2 in pcds[1:]:

        posLocalTrans = registerLocalCloud(cloud1, cloud2)

        detectTransLoop = np.dot(posLocalTrans, detectTransLoop)

        posWorldTrans =  np.dot(posWorldTrans, posLocalTrans)

        cloud1 = copy.deepcopy(cloud2)
        cloud2.transform(posWorldTrans)
        
        cloud_base = cloud_base + cloud2
        
        # downsampling
        # cloud_base.voxel_down_sample(voxel_size)

    o3d.visualization.draw_geometries([cloud_base])
    cl, ind = cloud_base.remove_statistical_outlier(nb_neighbors=30, std_ratio=3.0)
    cloud_base = cloud_base.select_by_index(ind)
    o3d.visualization.draw_geometries([cloud_base])

    # estimate normals
    # cloud_base = cloud_base.voxel_down_sample(voxel_size)
    # cloud_base.estimate_normals()
    
    # cloud_base.orient_normals_to_align_with_direction()

    # surface reconstruction
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud_base, depth=9, n_threads=1)[0]
    # mesh, des = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud_base, depth=15)

    

    # cloud_base.estimate_normals()
    
    print('Create 3d mesh use alpha shape')
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(cloud_base, 0.001)
    # mesh.compute_vertex_normals()

    cloud_base.compute_convex_hull()
    cloud_base.estimate_normals()
    cloud_base.orient_normals_consistent_tangent_plane(10)

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud_base, depth=10, scale=1.1, linear_fit=False)[0]
    bbox = cloud_base.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bbox)
    
    mesh.compute_vertex_normals()
    # mesh.paint_uniform_color([0.5, 0.5, 0.5])
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_duplicated_vertices()
    o3d.visualization.draw_geometries([cloud_base, mesh], mesh_show_back_face=True)

    # Visualize the mesh
    print('Visualize the mesh')
    # o3d.visualization.draw_geometries([mesh])

    # Save point cloud & mesh
    print('Visualize the mesh')
    o3d.io.write_point_cloud('./360degree_pointclouds/{0}/mesh/merged_pointclouds.ply'.format(dir_name), cloud_base)
    o3d.io.write_triangle_mesh('./360degree_pointclouds/{0}/mesh/3d_model.gltf'.format(dir_name), mesh)