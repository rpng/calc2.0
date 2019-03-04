loops_fn = 'kitti_loops.txt';
traj_fn = 'kitti_traj.txt';

loops = readtable(loops_fn);
x0 = loops{:,1};
y0 = loops{:,2};
x1 = loops{:,3};
y1 = loops{:,4};


traj = readtable(traj_fn);
ids = traj{:,1};
x = traj{:,2};
y = traj{:,3};


plot3(x, y, ids, 'b', 'linewidth', 2)
xlabel('x (m)')
ylabel('y (m)')
zlabel('Frame ID')
hold on;

for i = 1:length(x0)
    [~,id0] = min(sum([x0(i)-x y0(i)-y].^2, 2));
    [~,id1] = min(sum([x1(i)-x y1(i)-y].^2, 2));

    plot3([x0(i); x1(i)], [y0(i); y1(i)], [id0; id1], 'r', 'linewidth', 2)
end
