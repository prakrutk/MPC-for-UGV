import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class MoveHusky(Node):
    def _init_(self):
        super()._init_('move_husky')
        self.pubvel = Node.create_publisher(self, Twist, "/husky_velocity_controller/cmd_vel", 10)
        self.timer_ = self.create_timer(0.5, self.timer_callback)
        self.i = 0
    
    def move_husky(self):
        cmd_vel = Twist()

        # Fill in the fields of the Twist object (Note: cmd_vel is in base/body frame)
        # Linear velocity in the x-axis.
        cmd_vel.linear.x = 1.0
        cmd_vel.linear.y = 0.
        cmd_vel.linear.z = 0.

        # Angular velocity in the z-axis.
        cmd_vel.angular.x = 0.
        cmd_vel.angular.y = 0.
        cmd_vel.angular.z = 0

        self.cmd_vel_pub.publish(cmd_vel)

def main(args=None):

    rclpy.init(args=args)

    vel_publisher = MoveHusky()
    
    try:
        rclpy.spin(vel_publisher)
    except SystemExit:                 # <--- process the exception 
        rclpy.logging.get_logger("Quitting turtle1").info('Done')

    vel_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()