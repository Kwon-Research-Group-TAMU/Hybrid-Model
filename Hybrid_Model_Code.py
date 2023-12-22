from tensorflow import keras
import tensorflow as tf
tf.config.run_functions_eagerly(True)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tensorflow.keras import layers
import os
import subprocess
import pickle

'''Function to generate the spatiotemporal visual profiles for Diffusivity'''
def generate_video(img,t_array):
    filename='Training_Diffusivity'
    working_directory=os.getcwd()
    folder=os.path.join(working_directory, filename)
    for i in range(np.shape(img)[0]):
        plt.imshow(img[i], cmap='hot')
        plt.colorbar()
        plt.title(f"time in hours {t_array[i]}")
        plt.axis('off')
        plt.savefig(folder + "/file%02d.png" % i)
        plt.clf()
    os.chdir(folder)
    if(os.path.exists(os.path.join(folder,'video_name.mp4'))):
        os.remove(os.path.join(folder,'video_name.mp4'))

    subprocess.call([
        'ffmpeg', '-framerate', '8', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
        'video_name.mp4'
    ])
    os.chdir(working_directory)

'''Function to generate the spatiotemporal visual profiles for Cell Density'''
def generate_video_predictions(img,t_array):
    sub_folder_name='Training_Cell_Density'
    working_directory=os.getcwd()
    folder=os.path.join(working_directory, sub_folder_name)
    for i in range(np.shape(img)[0]):
        plt.imshow(img[i], cmap='hot')
        plt.colorbar()
        plt.title(f"time in hours {t_array[i]}")
        plt.axis('off')
        plt.savefig(folder + "/file%02d.png" % i)
        plt.clf()
    os.chdir(folder)
    if(os.path.exists(os.path.join(folder, 'video_name.mp4'))):
        os.remove(os.path.join(folder, 'video_name.mp4'))

    subprocess.call([
        'ffmpeg', '-framerate', '8', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
        'video_name.mp4'
    ])
    os.chdir(working_directory)


def generate_sequences(array_1,window_length):                  # Inputs=[N*H*W]=[B*H*W*W]
    for i in range(0,np.shape(array_1)[0]-window_length):
        if(i==0):
           sequential_input_array=array_1[i:window_length+i,:,:]
           for j in range(np.shape(sequential_input_array)[0]):
               if(j==0):
                   temp_array_1=sequential_input_array[j,:,:]
                   temp_array_1=np.reshape(temp_array_1,(np.shape(temp_array_1)[0],np.shape(temp_array_1)[1],1))
               else:
                   temp_array_2=sequential_input_array[j,:,:]
                   temp_array_1=np.concatenate((temp_array_1,np.reshape(temp_array_2,(np.shape(temp_array_2)[0],np.shape(temp_array_2)[1],1))),axis=-1)
           sequential_input_array=np.reshape(temp_array_1,(1,temp_array_1.shape[0],temp_array_1.shape[1],temp_array_1.shape[2]))

        else:
            temp_array=array_1[i:window_length+i,:,:]
            for j in range(np.shape(temp_array)[0]):
               if(j==0):
                   temp_array_1=temp_array[j,:,:]
                   temp_array_1=np.reshape(temp_array_1,(np.shape(temp_array_1)[0],np.shape(temp_array_1)[1],1))
               else:
                   temp_array_2=temp_array[j,:,:]
                   temp_array_1=np.concatenate((temp_array_1,np.reshape(temp_array_2,(np.shape(temp_array_2)[0],np.shape(temp_array_2)[1],1))),axis=-1)
            sequential_input_array=np.concatenate((sequential_input_array,np.reshape(temp_array_1,(1,temp_array_1.shape[0],temp_array_1.shape[1],temp_array_1.shape[2]))),axis=0)

    return sequential_input_array


'''The custom loss for Hybrid-Model training'''
class Customloss(tf.keras.losses.Loss):
    def __init__(self,multiplication_factor=10**6):
        super().__init__()
        self.multiplication_factor = multiplication_factor           # This is the factor by which the losses are multiplied

    def call(self,y_pred,y_true):
        loss = tf.reduce_mean(tf.square(y_true - y_pred))
        loss=loss*self.multiplication_factor*10**(3)*0.5
        return loss


'''The custom loss for Branch 1 pretraining'''
class Customloss2(tf.keras.losses.Loss):
    def __init__(self, grid_size=100):
        super().__init__()
        self.grid_size= grid_size

    def call(self, y_pred, y_true):
        loss1=tf.reduce_mean(tf.square(y_true - y_pred))
        return loss1


'''The custom loss for Branch 2 pretraining'''
class Customloss3(tf.keras.losses.Loss):
    def __init__(self,grid_size=100):
        super().__init__()
        self.grid_size = grid_size           # This is the factor by which the losses are multiplied

    def call(self,y_pred,y_true):
        loss2=tf.reduce_mean(tf.square(y_true[:,-2:-1] - y_pred[:,-2:-1]))
        loss3=tf.reduce_mean(tf.square(y_true[:,-1] - y_pred[:,-1]))
        loss=loss2+loss3
        return loss


class DifferentialEquation1(layers.Layer):
        def __init__(self,grid_size=100,time_steps_stacked=5,time_steps_simulated=6,Batch_size=64):
            super(DifferentialEquation1, self).__init__()
            # Create the grid
            self.grid_size=grid_size
            self.time_steps_stacked=time_steps_stacked
            self.time_steps_simulated=time_steps_simulated
            placeholder=tf.zeros([100,self.grid_size,self.grid_size])
            self.output1=tf.Variable(initial_value=placeholder, trainable=False, validate_shape=False)
            self.batch_size = Batch_size
            p1 = self.batch_size
            p2 = 100
            variable_size = (p1, p2, p2, 1)
            self.u_first_derivative_x_f =tf.Variable(name='u_first_derivative_x_f', initial_value=tf.zeros(variable_size),trainable=False)
            self.D_u_first_derivative_x_f=tf.Variable(name='D_u_first_derivative_x_f', initial_value=tf.zeros(variable_size),trainable=False)
            self.u_first_derivative_x_b = tf.Variable(name='u_first_derivative_x_b', initial_value=tf.zeros(variable_size),trainable=False)
            self.D_u_first_derivative_x_b=tf.Variable(name='D_u_first_derivative_x_b',initial_value=tf.zeros(variable_size),trainable=False)
            self.u_second_derivative_x=tf.Variable(name='u_second_derivative_x',initial_value=tf.zeros(variable_size),trainable=False)
            self.u_first_derivative_y_f=tf.Variable(name='u_first_derivative_y_f',initial_value=tf.zeros(variable_size),trainable=False)
            self.D_u_first_derivative_y_f=tf.Variable(name='D_u_first_derivative_y_f', initial_value=tf.zeros(variable_size),trainable=False)
            self.u_first_derivative_y_b=tf.Variable(name='u_first_derivative_y_b', initial_value=tf.zeros(variable_size),trainable=False)
            self.D_u_first_derivative_y_b=tf.Variable(name='D_u_first_derivative_y_b', initial_value=tf.zeros(variable_size),trainable=False)
            self.u_second_derivative_y = tf.Variable(name='u_second_derivative_y',initial_value=tf.zeros(variable_size), trainable=False)
            self.laplacian=tf.Variable(name='Laplacian', initial_value=tf.zeros(variable_size),trainable=False)
            self.grid_variable=tf.Variable(name='grid_variable',initial_value=tf.zeros(variable_size),trainable=False)
            self.a1=tf.Variable(name='a1',initial_value=tf.zeros(variable_size),trainable=False)
            self.b1=tf.Variable(name='b1', initial_value=tf.zeros(variable_size),trainable=False)
            self.c1=tf.Variable(name='c1', initial_value=tf.zeros(variable_size),trainable=False)
            self.du=tf.Variable(name='du', initial_value=tf.zeros(variable_size), trainable=False)
            self.Diffusivity=tf.Variable(name='Diffusivity_Saved', initial_value=tf.zeros(variable_size), trainable=False)
            self.Du_1=tf.Variable(name='Diffusivity', initial_value=tf.zeros(variable_size), trainable=False)
            self.Du_radial=tf.Variable(name='Diffusivity', initial_value=tf.zeros(variable_size), trainable=False)

        def build(self,input_shape):
           '''The build function used to define all the spatial discretization operators'''
           dx = 10
           dy = 10
           self.conv_a=tf.keras.layers.Conv2D(input_shape=(self.grid_size, self.grid_size, 1), filters=1,
                               kernel_size=(3, 3), strides=1,
                               padding='SAME',
                               kernel_initializer=tf.constant_initializer([0, 0, 0, -1 / dx, 1 / dx, 0, 0, 0, 0]),
                               trainable=False)
           self.conv_b=tf.keras.layers.Conv2D(input_shape=(self.grid_size,self.grid_size,1),
                                                                    filters=1, kernel_size=(3,3), strides=1,
                                                                    padding='SAME',
                                                                    kernel_initializer=tf.constant_initializer([[0,0,0,0,-1/dx,1/dx,0,0,0]]),
                                                                    trainable=False)

           self.conv_c=tf.keras.layers.Conv2D(input_shape=(self.grid_size, self.grid_size, 1),
                                                                    filters=1, kernel_size=(3, 3), strides=1,
                                                                    padding='SAME',
                                                                    kernel_initializer=tf.constant_initializer(
                                                                        [0,-1/dy,0,0,1/dy,0,0,0,0]),
                                                                    trainable=False)
           self.conv_d=tf.keras.layers.Conv2D(input_shape=(self.grid_size, self.grid_size, 1),
                                                                    filters=1, kernel_size=(3, 3), strides=1,
                                                                    padding='SAME',
                                                                    kernel_initializer=tf.constant_initializer(
                                                                        [0,0,0,0,-1/dy,0,0,1/dy,0]),
                                                                    trainable=False)

           self.Dx_forward=tf.keras.layers.Conv2D(input_shape=(self.grid_size,self.grid_size,1), filters=1,kernel_size=(3,3),strides=1,
                                                  padding='SAME',kernel_initializer=tf.constant_initializer([0,0,0,0,1,1,0,0,0]),
                                                  trainable=False)

           self.Dx_backward=tf.keras.layers.Conv2D(input_shape=(self.grid_size,self.grid_size,1), filters=1,kernel_size=(3,3),strides=1,
                                                  padding='SAME',kernel_initializer=tf.constant_initializer([0,0,0,1,1,0,0,0,0]),
                                                  trainable=False)

           self.Dy_forward=tf.keras.layers.Conv2D(input_shape=(self.grid_size,self.grid_size,1), filters=1,kernel_size=(3,3),strides=1,
                                                  padding='SAME',kernel_initializer=tf.constant_initializer([0,0,0,0,1,0,0,1,0]),
                                                  trainable=False)

           self.Dy_backward=tf.keras.layers.Conv2D(input_shape=(self.grid_size,self.grid_size,1), filters=1,kernel_size=(3,3),strides=1,
                                                  padding='SAME',kernel_initializer=tf.constant_initializer([0,1,0,0,1,0,0,0,0]),
                                                  trainable=False)

           self.x_central=tf.keras.layers.Conv2D(input_shape=(self.grid_size,self.grid_size,1), filters=1, kernel_size=(3,3), strides=1,
                                                 padding='SAME', kernel_initializer=tf.constant_initializer([0,0,0, -1/2,0,1/2, 0,0,0]), trainable=False)

           self.y_central=tf.keras.layers.Conv2D(input_shape=(self.grid_size,self.grid_size,1),filters=1, kernel_size=(3,3), strides=1,
                                                 padding='SAME', kernel_initializer=tf.constant_initializer([0,-1/2,0,0,0,0,0,1/2,0]), trainable=False)


        def call(self,inputs):

            '''The call function that is used to do the forward pass in tensorflow and also get the gradients while doing the backpropagtion'''

            time_step_initial=initial_bias*self.time_steps_simulated
            all_outputs=[]
            grid_var2 = inputs[:,0:self.grid_size*self.grid_size]
            grid_var2 = layers.Reshape((self.grid_size, self.grid_size, 1))(grid_var2)
            grid_var=grid_var2

            '''Separating the flattened layers to get the inputs for corresponding branches'''
            Du=inputs[:,self.grid_size*self.grid_size:2*self.grid_size*self.grid_size]
            Du=layers.Reshape((self.grid_size,self.grid_size,1))(Du)
            Du_saved=Du
            self.Diffusivity=Du_saved                                                                                   # tensor size: [batch, H, W,1]

            K=inputs[:,2*self.grid_size*self.grid_size]
            layer_reshape=layers.Reshape((1,1))(K)
            layer1=layers.Lambda(lambda x: tf.matmul(x,tf.ones([1,self.grid_size*self.grid_size])))(layer_reshape)
            layer_respahe2=layers.Reshape((self.grid_size,self.grid_size,1))(layer1)
            K=layer_respahe2

            # Reshaping the Lambda inputs to obtain the reshaped outputs of size=(None, grid_size, grid_size, 1)
            Lambda1=inputs[:,2*self.grid_size*self.grid_size+1]
            layers_reshape=layers.Reshape((1,1))(Lambda1)
            layer1=layers.Lambda(lambda x: tf.matmul(x,tf.ones([1,self.grid_size*self.grid_size])))(layers_reshape)
            layer_reshape2=layers.Reshape((self.grid_size,self.grid_size,1))(layer1)
            Lambda1=layer_reshape2

            '''In the following lines the components used in build are used for developing all the operations needed to solve the Advection-Diffusion equation with the Forward Euler's methos'''
            dx=10
            dy=10

            '''Deriving the radial gradient of Diffusivity '''
            x_grid=tf.constant([(a+10**(-3))*dx for a in range(-int(self.grid_size/2),int(self.grid_size/2))], dtype=tf.float32)
            y_grid=tf.constant([(a+10**(-3))*dx for a in range(-int(self.grid_size/2), int(self.grid_size/2))], dtype=tf.float32)
            x_t, y_t=tf.meshgrid(x_grid, y_grid)

            magnitude=tf.math.sqrt(tf.square(x_t)+tf.square(y_t))

            sin_matrix=tf.divide(y_t,magnitude)                                                                         #[H, W]

            cos_matrix=tf.divide(x_t, magnitude)                                                                        #[H, W]

            Dc_dx=self.x_central(self.Diffusivity)/dx                                                                   #[B, H, W, 1]
            Dc_dx=layers.Reshape((self.grid_size, self.grid_size))(Dc_dx)                                               #[B, H, W]

            Dc_dy=self.y_central(self.Diffusivity)/dy                                                                   #[B, H, W, 1]
            Dc_dy=layers.Reshape((self.grid_size, self.grid_size))(Dc_dy)                                               #[B, H, W]

            Du_radial_1=tf.keras.layers.Lambda(lambda x: x[0]*x[1])([Dc_dx,cos_matrix])                                   # [B, H, W]
            Du_radial_2=tf.keras.layers.Lambda(lambda x: x[0]*x[1])([Dc_dy, sin_matrix])                                  # [B, H, W]

            self.Du_radial=Du_radial_1+Du_radial_2

            '''The function has Nuemann boundary condition as of now, for both Drichlet and Nuemann the same boundary condition gets applies, a user can write a class for Drichilet Boundary conditions and update the code'''
            BC_type='Nuemann'
            delta_t=0.015

            # Apply BC to the initial set
            grid_var_left_all=[]                                     # Counter for the number of stacks present in the system
            grid_var_right_all=[]
            grid_var_top_all=[]
            grid_var_down_all=[]

            grid_var_all=[]
            grid_var_all.append(grid_var)

            if(BC_type=='Dirichilet'):
                grid_var_left_all.append(Nueman_Bounday('left',100)(grid_var))
                grid_var_right_all.append(Nueman_Bounday('right', 100)(grid_var_left_all[0]))
                grid_var_top_all.append(Nueman_Bounday('top', 100)(grid_var_right_all[0]))
                grid_var_down_all.append(Nueman_Bounday('down', 100)(grid_var_top_all[0]))

            else:
                grid_var_left_all.append(Nueman_Bounday('left',100)(grid_var))
                grid_var_right_all.append(Nueman_Bounday('right', 100)(grid_var_left_all[0]))
                grid_var_top_all.append(Nueman_Bounday('top', 100)(grid_var_right_all[0]))
                grid_var_down_all.append(Nueman_Bounday('down', 100)(grid_var_top_all[0]))

            time_array=[]
            all_grid_sol=[]
            all_grid_sol.append(grid_var)
            Du_1 = Du
            self.grid_variable=all_grid_sol[-1]

            truth_value_1=tf.equal(self.grid_variable,grid_var_down_all[-1])
            truth_value=tf.reduce_sum(tf.cast(truth_value_1,tf.float32))

            self.Du_1=Du_1
            tt=2                                             # For the case when the Boundary conditions are applied
            count_steps=-1
            for step in range(self.time_steps_simulated+time_step_initial):

               if(step>=time_step_initial):
                   count_steps+=1

               if(tt!=1):
                    self.u_first_derivative_x_f=self.conv_a(self.grid_variable)

                    self.D_first_derivative_x_f=self.Dx_forward(self.Du_1)

                    self.D_u_first_derivative_x_f=tf.keras.layers.Lambda(lambda x: tf.multiply(x[0],x[1]))([self.D_first_derivative_x_f,self.u_first_derivative_x_f])

                    self.u_first_derivative_x_b=self.conv_b(self.grid_variable)

                    self.D_first_derivative_x_b=self.Dx_backward(self.Du_1)

                    self.D_u_first_derivative_x_b=tf.keras.layers.Lambda(lambda x: tf.multiply(x[0],x[1]))([self.D_first_derivative_x_b,self.u_first_derivative_x_b])

                    self.u_second_derivative_x=tf.keras.layers.Lambda(lambda x: x[0]/(2*dx)-x[1]/(2*dx))([self.D_u_first_derivative_x_f,self.D_u_first_derivative_x_b])

                    self.u_first_derivative_y_f=self.conv_c(self.grid_variable)

                    self.D_first_derivative_y_f=self.Dy_forward(self.Du_1)

                    self.D_u_first_derivative_y_f=tf.keras.layers.Lambda(lambda x: tf.multiply(x[0],x[1]))([self.D_first_derivative_y_f,self.u_first_derivative_y_f])

                    self.u_first_derivative_y_b=self.conv_d(self.grid_variable)

                    self.D_first_derivative_y_b=self.Dy_backward(self.Du_1)

                    self.D_u_first_derivative_y_b=tf.keras.layers.Lambda(lambda x: tf.multiply(x[0], x[1]))([self.D_first_derivative_y_b,self.u_first_derivative_y_b])

                    self.u_second_derivative_y=tf.keras.layers.Lambda(lambda x: x[0] / (2*dy) - x[1] / (2*dy))([self.D_u_first_derivative_y_f, self.D_u_first_derivative_y_b])

                    self.laplacian=tf.keras.layers.Lambda(lambda x: x[0]+x[1])([self.u_second_derivative_x,self.u_second_derivative_y])

                    # Apply Boundary conditions

                    if(BC_type=='Dirichilet'):

                        self.a1=tf.keras.layers.Lambda(lambda x: 1 - tf.math.divide_no_nan(x[0], x[1]))([self.grid_variable, K])

                        self.b1=tf.keras.layers.Lambda(lambda x: tf.multiply(x[0], x[1]))([self.grid_variable, self.a1])

                        self.c1=tf.keras.layers.Lambda(lambda x: tf.multiply(x[0], x[1]))([Lambda1, self.b1])

                        self.du=tf.keras.layers.Lambda(lambda x: x[0] + x[1])([self.laplacian, self.c1])

                        self.grid_variable=keras.layers.Lambda(lambda x: x[0] + x[1] * delta_t)([self.grid_variable, self.du])

                        self.grid_variable=Nueman_Bounday('left', 100)(self.grid_variable)

                        self.grid_variable=Nueman_Bounday('right', 100)(self.grid_variable)

                        self.grid_variable = Nueman_Bounday('top', 100)(self.grid_variable)

                        self.grid_variable=Nueman_Bounday('down', 100)(self.grid_variable)
                    else:
                        # a1=1-tf.math.divide_no_nan(u,K)

                        self.a1=tf.keras.layers.Lambda(lambda x: 1 - tf.math.divide_no_nan(x[0], x[1]))([self.grid_variable, K])

                        self.b1 = tf.keras.layers.Lambda(lambda x: tf.multiply(x[0], x[1]))([self.grid_variable, self.a1])

                        self.c1 = tf.keras.layers.Lambda(lambda x: tf.multiply(x[0], x[1]))([Lambda1, self.b1])

                        self.du = tf.keras.layers.Lambda(lambda x: x[0] + x[1])([self.laplacian, self.c1])

                        self.grid_variable = keras.layers.Lambda(lambda x: x[0] + x[1] * delta_t)([self.grid_variable, self.du])

                        self.grid_variable = Nueman_Bounday('left', 100)(self.grid_variable)

                        self.grid_variable = Nueman_Bounday('right', 100)(self.grid_variable)

                        self.grid_variable = Nueman_Bounday('top', 100)(self.grid_variable)

                        self.grid_variable = Nueman_Bounday('down', 100)(self.grid_variable)
               else:

                   self.u_first_derivative_x_f=self.conv_a(self.grid_variable)

                   self.D_first_derivative_x_f=self.Dx_forward(self.Du_1)

                   self.D_u_first_derivative_x_f=tf.keras.layers.Lambda(lambda x: tf.multiply(x[0],x[1]))([self.D_first_derivative_x_f,self.u_first_derivative_x_f])

                   self.u_first_derivative_x_b=self.conv_b(self.grid_variable)

                   self.D_first_derivative_x_b=self.Dx_backward(self.Du_1)

                   self.D_u_first_derivative_x_b=tf.keras.layers.Lambda(lambda x: tf.multiply(x[0],x[1]))([self.D_first_derivative_x_b,self.u_first_derivative_x_b])

                   self.u_second_derivative_x=tf.keras.layers.Lambda(lambda x: x[0]/(2*dx)-x[1]/(2*dx))([self.D_u_first_derivative_x_f,self.D_u_first_derivative_x_b])

                   self.u_first_derivative_y_f=self.conv_c(self.grid_variable)

                   self.D_first_derivative_y_f=self.Dy_forward(self.Du_1)

                   self.D_u_first_derivative_y_f=tf.keras.layers.Lambda(lambda x: tf.multiply(x[0],x[1]))([self.D_first_derivative_y_f,self.u_first_derivative_y_f])

                   self.u_first_derivative_y_b=self.conv_d(self.grid_variable)

                   self.D_first_derivative_y_b=self.Dy_backward(self.Du_1)

                   self.D_u_first_derivative_y_b=tf.keras.layers.Lambda(lambda x: tf.multiply(x[0], x[1]))([self.D_first_derivative_y_b,self.u_first_derivative_y_b])

                   self.u_second_derivative_y=tf.keras.layers.Lambda(lambda x: x[0] / (2*dy) - x[1] / (2*dy))([self.D_u_first_derivative_y_f, self.D_u_first_derivative_y_b])

                   self.laplacian=tf.keras.layers.Lambda(lambda x: x[0]+x[1])([self.u_second_derivative_x,self.u_second_derivative_y])

                   self.a1=tf.keras.layers.Lambda(lambda x: 1 - tf.math.divide_no_nan(x[0], x[1]))([self.grid_variable, K])

                   self.b1=tf.keras.layers.Lambda(lambda x: tf.multiply(x[0], x[1]))([self.grid_variable, self.a1])

                   self.c1=tf.keras.layers.Lambda(lambda x: tf.multiply(x[0], x[1]))([Lambda1, self.b1])

                   self.du=tf.keras.layers.Lambda(lambda x: x[0] + x[1])([self.laplacian, self.c1])

                   if(step>=0 and step<=time_step_initial+self.time_steps_simulated):
                      self.grid_variable = keras.layers.Lambda(lambda x: x[0] + x[1] * delta_t)([self.grid_variable, self.du])

               # Concatenation of the steps obtained

               if(count_steps%self.time_steps_stacked==0 and count_steps>0):
                        time_array.append(delta_t*step)
                        #all_outputs.append(self.Du_1)
                        all_outputs.append(self.grid_variable)

            return keras.layers.concatenate(all_outputs,axis=-1)

class Nueman_Bounday(layers.Layer):
        def __init__(self,Boundary='left',grid_size=100):
            super(Nueman_Bounday, self).__init__()
            self.boundary=Boundary
            self.grid_size=grid_size

        def call(self,inputs):
            if(self.boundary=='left'):
                left_boundary_tensor=inputs[:,:,0,:]
                all_other_values=inputs[:,:,2:,:]
                first_concatenation=tf.keras.layers.concatenate([left_boundary_tensor,left_boundary_tensor],axis=2)
                reshaped_layer=tf.keras.layers.Reshape((self.grid_size,2,1))(first_concatenation)
                second_concatenation=tf.keras.layers.concatenate([reshaped_layer,all_other_values],axis=2)
                return second_concatenation
            elif(self.boundary=='right'):
                right_boundary_tensor=inputs[:,:,-1,:]
                all_other_values=inputs[:,:,0:self.grid_size-2,:]
                first_concatenation=tf.keras.layers.concatenate([right_boundary_tensor,right_boundary_tensor],axis=2)
                reshaped_layer=tf.keras.layers.Reshape((self.grid_size,2,1))(first_concatenation)
                second_concatenation=tf.keras.layers.concatenate([all_other_values,reshaped_layer],axis=2)
                return second_concatenation
            elif(self.boundary=='top'):
                top_boundary_tensor=inputs[:,0,:,:]
                all_other_values=inputs[:,2:,:,:]
                first_concatenation=tf.keras.layers.concatenate([top_boundary_tensor,top_boundary_tensor],axis=1)
                reshaped_layer=tf.keras.layers.Reshape((2,self.grid_size,1))(first_concatenation)
                second_concatenation=tf.keras.layers.concatenate([reshaped_layer,all_other_values],axis=1)
                return second_concatenation
            elif(self.boundary=='down'):
                bottom_boundary_tensor=inputs[:,-1,:,:]
                all_other_values=inputs[:,0:self.grid_size-2,:,:]
                first_concatenation=tf.keras.layers.concatenate([bottom_boundary_tensor,bottom_boundary_tensor],axis=1)
                reshaped_layer=tf.keras.layers.Reshape((2,self.grid_size,1))(first_concatenation)
                second_concatenation=tf.keras.layers.concatenate([all_other_values,reshaped_layer],axis=1)
                return second_concatenation



'''The hybrid model training parameters are fixed in this part of the code'''

pretrain=False
delta_t=0.015
dx=10
input_height=100
input_width=100
grid_size=100
learning_Rate=1e-4
window_length=16                                              # The window length for stacking the number of input frames
initial_bias=0                                                # The initial bias is used when the starting point needs to be shifted
time_step_initial=int(window_length)-initial_bias-1
number_channels=window_length
common_input_shape=[input_height,input_width,number_channels]
Common_input_layer=tf.keras.Input(shape=common_input_shape)

'''The minimum and maximum values of Diffusivity, lambda, and K'''
min_Du=0; max_Du=1594
min_lambda=0.0317; max_lambda=2*0.0317
min_K=0.00258/2;     max_K=0.00258
min_c=0;     max_c=0.0012003053

'''Branch 1'''
Input_initial=layers.Lambda(lambda x: x[:,:,:,time_step_initial])(Common_input_layer)
Input_initial_reshape=layers.Reshape((grid_size,grid_size,1))(Input_initial)
conv1D=layers.Conv1D(filters=1,kernel_size=1,kernel_initializer=tf.keras.initializers.ones(),activation=None,trainable=False)(Input_initial_reshape)
flatten1=layers.Flatten()(conv1D)
flatten1_scaled=layers.Lambda(lambda x: x*(max_c-min_c)+min_c)(flatten1)

'''Branch 2'''
Conv3D_1=layers.Conv2D(window_length, (3,3), padding='same',activation='relu')(Common_input_layer)
Conv2D_2=layers.Conv2D(80,(3,3), padding='same', activation='relu')(Conv3D_1)
Conv2D_3=layers.Conv2D(90,(3,3), padding='same', activation='relu')(Conv2D_2)
Conv2D_4=layers.Conv2D(90,(3,3), padding='same', activation='relu')(Conv2D_3)
Conv2D_5=layers.Conv2D(80,(3,3), padding='same', activation='relu')(Conv2D_4)
Conv2D_6=layers.Conv2D(1,(3,3),  padding='same',activation='relu')(Conv2D_5)
flatten2=layers.Flatten()(Conv2D_6)
flatten2_scaled=layers.Lambda(lambda x: x*(max_Du-min_Du)+min_Du)(flatten2)

'''Branch 3'''
Conv3D_7=layers.Conv2D(window_length, (10,10), padding='same',activation='relu')(Common_input_layer)
Conv3D_8=layers.Conv2D(1, (10,10), padding='same',activation='relu')(Conv3D_7)
Conv2D_7=layers.Conv2D(1, (10,10), strides=(2,2), padding='valid',activation='relu')(Conv3D_7)
Conv2D_8=layers.Conv2D(1,(10,10), strides=(2,2), padding='valid', activation='relu')(Conv2D_7)
Conv2D_9=layers.Conv2D(1,(9,9), strides=(2,2),padding='valid', activation='relu')(Conv2D_8)
Conv2D_10=layers.Flatten()(Conv2D_9)
Dense1=layers.Dense(96,activation='relu')(Conv2D_10)
Dense2=layers.Dense(16,activation='relu')(Dense1)
Dense3=layers.Dense(16,activation='sigmoid')(Dense2)
Dense4=layers.Dense(16,activation='linear')(Dense3)
Dense5=layers.Dense(2,activation='relu')(Dense4)
Dense_K=layers.Lambda(lambda x: x[:,0]*(max_K-min_K)+min_K)(Dense5)
Dense_Lambda=layers.Lambda(lambda x: x[:,1]*(max_lambda-min_lambda)+min_lambda)(Dense5)
Dense_K_Reshaped=layers.Reshape((1,))(Dense_K)
Dense_Lambda_Reshaped=layers.Reshape((1,))(Dense_Lambda)
Dense6=layers.Concatenate()([Dense_K_Reshaped,Dense_Lambda_Reshaped])

'''Concatenated layers to make the complete model'''
concatenated_layer=layers.concatenate([flatten1_scaled,flatten2_scaled,Dense6])
output=DifferentialEquation1(100,9,10,96)(concatenated_layer)
output_rescaled=layers.Lambda(lambda x: x/(max_c-min_c)-min_c/(max_c-min_c))(output)
model=tf.keras.Model(inputs=Common_input_layer,outputs=output_rescaled)

'''Loading the pretraining data'''
filename='Pretraining_data_Du'
fileObject=open(filename,'rb')
Du_pre_training=pickle.load(fileObject)
fileObject.close()

filename='Pretraining_data_K'
fileObject=open(filename,'rb')
K_all=pickle.load(fileObject)
fileObject.close()

filename='Pretraining_data_lambda'
fileObject=open(filename,'rb')
Lambda_all=pickle.load(fileObject)
fileObject.close()

'''Loading the training data'''
filename='Training_data'
fileObject=open(filename,'rb')
grid_data=pickle.load(fileObject)
fileObject.close()

filename='Time_data'
fileObject=open(filename,'rb')
time_taken=pickle.load(fileObject)
fileObject.close()

'''Normalizing the training input data'''
x_train_1=grid_data.copy()
for i in range(np.shape(x_train_1)[0]):
    for j in range(np.shape(x_train_1)[1]):
        for k in range(np.shape(x_train_1)[2]):
            x_train_1[i,j,k]=(x_train_1[i,j,k]-min_c)/(max_c-min_c)

'''Generating the sequences of inputs->[Batch, grid_size, grid_size, window_size]'''
x_train=generate_sequences(x_train_1.copy(),window_length)

'''Getting the outputs for training->[Batch,1] and normalizing the final'''
filename='Y_train_data'
fileObject=open(filename,'rb')
y_train=pickle.load(fileObject)
fileObject.close()
y_train=(y_train-min_c)/(max_c-min_c)


'''Generating the complete data for pre-training'''

for i in range(np.shape(x_train)[0]):

    '''Inputs for pre train'''
    flatten_1_pretrain=x_train[i,:,:,time_step_initial].flatten()

    '''Pretrain Outputs from Branch 2'''
    flatten_2_pretrain=Du_pre_training[i+time_step_initial,0,:,:,0].flatten()
    flatten_2_pretrain=(flatten_2_pretrain-min_Du)/(max_Du-min_Du)

    '''Pretrain Outputs from Branch 3'''
    K_concatenate=np.array([(K_all[i+time_step_initial]-min_K)/(max_K-min_K)])
    Lambda_concatenate=np.array([(Lambda_all[i+time_step_initial]-min_lambda)/(max_lambda-min_lambda)])

    '''Concatenating the inputs'''
    pre_train_y_temp = np.concatenate((flatten_1_pretrain, flatten_2_pretrain, K_concatenate, Lambda_concatenate), axis=0)
    pre_train_y_temp_1=flatten_1_pretrain
    pre_train_y_temp_2=np.concatenate((K_concatenate,Lambda_concatenate))
    pre_train_test_1_temp=np.concatenate((grid_data[i+time_step_initial,:,:].flatten(), Du_pre_training[i+time_step_initial,0,:,:,0].flatten(), np.array([K_all[i+time_step_initial]]), np.array([Lambda_all[i+time_step_initial]])), axis=0)
    if(i==0):
        pre_train_y_data=np.array([pre_train_y_temp])
        pre_train_test_1=np.array([pre_train_test_1_temp])
        pre_train_y_1=np.array([pre_train_y_temp_1])
        pre_train_y_2=np.array([pre_train_y_temp_2])
    else:
        pre_train_y_data=np.concatenate((pre_train_y_data,np.array([pre_train_y_temp])),axis=0)
        pre_train_y_1=np.concatenate((pre_train_y_1, np.array([pre_train_y_temp_1])), axis=0)
        pre_train_y_2=np.concatenate((pre_train_y_2, np.array([pre_train_y_temp_2])), axis=0)
        pre_train_test_1=np.concatenate((pre_train_test_1,np.array([pre_train_test_1_temp])),axis=0)


'''This is the pre-training phase, the argument that is taken by the training function, i.e., pre-training to be true or false'''
'''Pre-training True: Implies that pre-training would be done'''
'''Pre-training False: Implies good initial weights and biases would be loaded'''


'''Making the model for Branch 1 '''
model_pretrain_branch1=tf.keras.Model(inputs=Common_input_layer, outputs=flatten2)
custom_loss_branch_1=Customloss2(grid_size)
optimizer_2=tf.optimizers.Adam(lr=learning_Rate)
model_pretrain_branch1.compile(optimizer=optimizer_2,loss=custom_loss_branch_1)

'''Training or Loading the weights to Branch 1'''
if(pretrain==True):
    model_pretrain_branch1.fit(x_train, pre_train_y_1, epochs=1, batch_size=64)
else:
    model_pretrain_branch1.load_weights('Branch_1.h5')

'''Making the model for Branch 2 '''
model_pretrain_branch2=tf.keras.Model(inputs=Common_input_layer, outputs=Dense5)
custom_loss_branch_2=Customloss3(grid_size)
optimizer_3=tf.optimizers.Adam(lr=learning_Rate)
model_pretrain_branch2.compile(optimizer=optimizer_3,loss=custom_loss_branch_2)

'''Training or Loading the weights to Branch 2'''
if(pretrain==True):
    model_pretrain_branch2.fit(x_train, pre_train_y_2, epochs=1, batch_size=64)
else:
    model_pretrain_branch2.load_weights('Branch_2.h5')

'''Assigning weights from pretrained Branch layer 1 to the actual model'''
for i in range(len(model_pretrain_branch1.layers)):
    for j in range(len(model.layers)):
        if(model_pretrain_branch1.layers[i].name==model.layers[j].name):
            source_weights=model_pretrain_branch1.layers[i].weights
            model.layers[j].set_weights(source_weights)
            print('After Pretraining assigned weights of layer:', model.layers[j].name)

'''Assigning weights from pretrained Branch layer 2 to the actual model'''
for i in range(len(model_pretrain_branch2.layers)):
    for j in range(len(model.layers)):
        if(model_pretrain_branch2.layers[i].name==model.layers[j].name):
            source_weights=model_pretrain_branch2.layers[i].weights
            model.layers[j].set_weights(source_weights)
            print('After Pretraining assigned weights of layer:', model.layers[j].name)


'''Compiling the hybrid-model and fitting the model with the required data'''
optimizer_3 = keras.optimizers.Adam(lr=learning_Rate)
custom_loss = Customloss(multiplication_factor=1e6)
model.compile(optimizer=optimizer_3, loss=custom_loss)
model.fit(x_train,y_train,epochs=10,batch_size=96)

'''Making predictions from the model and getting the spatiotemporal evolution of cell-density and getting the video '''
predicted_value=model.predict(x_train)
predicted_value=predicted_value*(max_c-min_c)+min_c
t_array=time_taken[window_length:len(time_taken)+1]
generate_video_predictions(predicted_value,t_array)

'''Developing the keras layer to get spatiotemporal evolution of diffusivity'''
model_layers_output_D=tf.keras.Model(inputs=Common_input_layer,outputs=Conv2D_6)
for i in range(len(model.layers)):
    for j in range(len(model_layers_output_D.layers)):
        if(model.layers[i].name==model_layers_output_D.layers[j].name):
            source_weights=model.layers[i].weights
            model_layers_output_D.layers[j].set_weights(source_weights)
            print('After training assigned weights of layer:', model.layers[j].name)


'''Predicting the spatiotemporal evolution of Diffusivity and generating the video'''
Diffusivity_output=model_layers_output_D.predict(x_train[:,:,:,:])
Diffusivity_output=np.reshape(Diffusivity_output,(Diffusivity_output.shape[0],Diffusivity_output.shape[1],Diffusivity_output.shape[2]))
Diffusivity_output=Diffusivity_output*(max_Du-min_Du)+min_Du
generate_video(Diffusivity_output,t_array)


'''Model layers to get the values of the values of K and Lambda '''
model_layers_output_K=tf.keras.Model(inputs=Common_input_layer,outputs=Dense6)
Rate_Constants_output=model_layers_output_K.predict(x_train[:,:,:,:])
plt.semilogy(t_array, Rate_Constants_output[:,0])
plt.semilogy(t_array, Rate_Constants_output[:,1])
plt.ylim([10**(-4),10**(-1)])
plt.legend(['K','Lambda'])
plt.xlabel('The value of time in hours')

'''Saving the temporal profiles of K and Lambda'''
df1=np.concatenate((np.array([t_array]), np.array([Rate_Constants_output[:,0]]), np.array([Rate_Constants_output[:,1]])),axis=0)
import pandas as pd
df1=pd.DataFrame(np.transpose(df1), columns=['Time (hr)', 'K', 'Lambda'])
df1.to_csv('K_lambda.csv')
