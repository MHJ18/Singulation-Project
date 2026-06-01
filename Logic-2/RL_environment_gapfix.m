classdef RL_environment_gapfix < rl.env.MATLABEnvironment
    %AMS_RL: Template for defining custom environment in MATLAB.    
    
    %% Properties (set properties' attributes accordingly)
    properties
        % Specify and initialize environment's necessary properties    
        v_treadmill = 1.90; % m/s

        d_AMS = 0.2; % m - size of the AMS
        n_i_AMS = 5; % numeber of AMS along y
        n_j_AMS = 5; % number of AMS along x
        n_actions_art = 2; % number of actions of each AMS

        upperlim_rot = pi*45/180; % upper limit rotation for AMS action
        upperlim_v = 2.2; % max velocity for AMS
        lowerlim_rot = - pi*45/180; % lower limit rotation for AMS action
        lowerlim_v = 0.5; % min velocity for AMS

        vy_boxes = [];
        x_boxes_prec = [];
        y_boxes_prec = [];

        index_AMS = [];

        LastAction = zeros(50,1);

        i_box = [];
        j_box = [];

        n_boxes_tot_vect = [];
        n_boxes_tot = 0;
        max_gen_boxes = 10;
        cont_new_box = 0;

        d_boxes = zeros(10,1);
        x_boxes = zeros(10,1);
        y_boxes = zeros(10,1);
        x_boxes_vect = zeros(10000,10);
        y_boxes_vect = zeros(10000,10);
        cont = 0;
        pack_exited = zeros(10,1);
        exit_order = zeros(10,1);
        exit_time = zeros(10,1);
        index_exit = 0;
        LastExitedCount = 0;

        toll_contatto = 0.005;
        desired_gap = 0.20; % m - vertical gap for one-by-one exit
        min_exit_time_gap = 0.20; % s - avoid parcels leaving together
        exit_zone_depth = 0.80; % m - watch spacing before AMS exit

        dt = 0.01; % s - timestep for simulation

        time = 0; % s - simulation time
    end

    properties (Hidden)
        % Flags for visualization
        VisualizeAnimation = true
        VisualizeActions = false
        VisualizeStates = true        
    end
    
    properties
        % Initialize system state [on1,x1,y1,on2,x2,y2,...,on25,x25,y25]'
        State = zeros(100,1)
    end
    
    properties(Access = protected)
        % Initialize internal flag to indicate episode termination
        IsDone = false        
    end

    properties (Transient, Access = private)
        Visualizer = []
    end

    %% Necessary Methods
    methods              
        % Contructor method creates an instance of the environment
        % Change class name and constructor name accordingly
        function this = RL_environment_gapfix()
            % Initialize Observation settings

            %================= OBSERVATION PART =================
            % states used by the agent
            ObservationInfo = rlNumericSpec([100 1]);
            ObservationInfo.Name = 'System States';
            ObservationInfo.Description = 'system states';

            %================= END OBSERVATION PART =============
            
            % Initialize Action settings
            n_actions = 5*5*2;
            ActionInfo = rlNumericSpec([n_actions 1]);
            ActionInfo.Name = 'AMS Action';
            ActionInfo.Description = 'r1, v1, r2, v2, ...';
            ActionInfo.LowerLimit = -ones(n_actions,1);
            ActionInfo.UpperLimit = ones(n_actions,1);
            
            % The following line implements built-in functions of RL env
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
        end
        
        % Apply system dynamics and simulates the environment with the 
        % given action for one step.
        function [Observation,Reward,IsDone,Info] = step(this,Action)
            Info = [];

            this.time = this.time + this.dt;

            this.cont = this.cont + 1;

            l_AMS_matrix = this.d_AMS*this.n_j_AMS; % m

            ActLimUp = zeros(50,1);
            ActLimLow = zeros(50,1);

            for ii=1:2:50
                ActLimUp(ii) = this.upperlim_rot;
                ActLimUp(ii+1) = this.upperlim_v;
                ActLimLow(ii) = this.lowerlim_rot;
                ActLimLow(ii+1) = this.lowerlim_v;
            end

            % Actions are normalized [0-1]
            % De-normalizing the actions
            AMS_actions = ActLimLow + (1 + Action) .* (ActLimUp - ActLimLow)./2;
            for ii=1:50
                AMS_actions(ii) = max(ActLimLow(ii),min(ActLimUp(ii),AMS_actions(ii)));
            end

            this.LastAction = AMS_actions;

            v_AMS = zeros(5,5);
            rotation_AMS = zeros(5,5);

            for ii = 1:this.n_i_AMS
                for jj = 1:this.n_j_AMS
                    % rotational actions
                    rotation_AMS(ii,jj) = AMS_actions(jj*2-1+(ii-1)*10);
                    % velocity actions
                    v_AMS(ii,jj) = AMS_actions(jj*2+(ii-1)*10);
                end
            end

            % new boxes generation each 0.75 s. 2 boxes are generated.

            if mod(round(this.time,2),0.75) == 0 && this.n_boxes_tot<this.max_gen_boxes && this.time>0.25
                
                this.cont_new_box = this.cont_new_box + 1;
                
                gen_n_boxes = 2;
                if gen_n_boxes>2
                    gen_n_boxes=2;
                end

                while this.n_boxes_tot+gen_n_boxes>this.max_gen_boxes
                    gen_n_boxes = gen_n_boxes-1;
                end

                this.n_boxes_tot = this.n_boxes_tot+gen_n_boxes;
                this.n_boxes_tot_vect(this.cont_new_box) = gen_n_boxes;

                for ii=1:gen_n_boxes

                    if gen_n_boxes>1
                        if ii == 1
                            this.x_boxes(this.n_boxes_tot-1) = this.d_boxes(this.n_boxes_tot-1)*0.5 + (this.d_AMS*2.5-this.d_boxes(this.n_boxes_tot-1)*0.5)*rand(1);
                            this.y_boxes(this.n_boxes_tot-1) = 0.001 + rand(1)*0.01;
                            this.x_boxes_prec(this.n_boxes_tot-1) = this.x_boxes(this.n_boxes_tot-1);
                            this.y_boxes_prec(this.n_boxes_tot-1) = this.y_boxes(this.n_boxes_tot-1);
                        else
                            this.x_boxes(this.n_boxes_tot) = this.d_AMS*2.5+this.d_boxes(this.n_boxes_tot)*0.5 + (l_AMS_matrix-(this.d_AMS*2.5+this.d_boxes(this.n_boxes_tot)*0.5))*rand(1);
                            this.y_boxes(this.n_boxes_tot) = 0.001 + rand(1)*0.05;
                            if abs(this.x_boxes(this.n_boxes_tot)-this.x_boxes(this.n_boxes_tot-1)) <= this.d_boxes(this.n_boxes_tot)/2+this.d_boxes(this.n_boxes_tot-1)/2
                                this.x_boxes(this.n_boxes_tot) = this.x_boxes(this.n_boxes_tot-1) + this.d_boxes(this.n_boxes_tot)/2 + this.d_boxes(this.n_boxes_tot-1)/2 + 0.1;
                            end
                            if this.x_boxes(this.n_boxes_tot)-this.d_boxes(this.n_boxes_tot)/2<0
                                this.x_boxes(this.n_boxes_tot) = this.d_boxes(this.n_boxes_tot)/2;
                            elseif this.x_boxes(this.n_boxes_tot)+this.d_boxes(this.n_boxes_tot)/2>l_AMS_matrix
                                this.x_boxes(this.n_boxes_tot) = l_AMS_matrix-this.d_boxes(this.n_boxes_tot)/2;
                            end
                            if abs(this.x_boxes(this.n_boxes_tot)-this.x_boxes(this.n_boxes_tot-1)) <= this.d_boxes(this.n_boxes_tot)/2+this.d_boxes(this.n_boxes_tot-1)/2
                                this.x_boxes(this.n_boxes_tot-1) = this.x_boxes(this.n_boxes_tot) - (this.d_boxes(this.n_boxes_tot)/2 + this.d_boxes(this.n_boxes_tot-1)/2 + 0.1);
                            end
                            this.x_boxes_prec(this.n_boxes_tot) = this.x_boxes(this.n_boxes_tot);
                            this.y_boxes_prec(this.n_boxes_tot) = this.y_boxes(this.n_boxes_tot);
                        end
                    else
                        this.x_boxes(this.n_boxes_tot) = this.d_boxes(this.n_boxes_tot)*0.55 + (l_AMS_matrix-this.d_boxes(this.n_boxes_tot)*0.55)*rand(1);
                        this.y_boxes(this.n_boxes_tot) = 0.001 + rand(1)*0.01;
                        this.x_boxes_prec(this.n_boxes_tot) = this.x_boxes(this.n_boxes_tot);
                        this.y_boxes_prec(this.n_boxes_tot) = this.y_boxes(this.n_boxes_tot);
                        this.x_boxes_vect(this.n_boxes_tot,this.cont-1) = this.x_boxes(this.n_boxes_tot);
                        this.y_boxes_vect(this.n_boxes_tot,this.cont-1) = this.y_boxes(this.n_boxes_tot);
                    end

                end

            end

            for ii=this.n_boxes_tot+1:this.max_gen_boxes
                this.x_boxes(ii) = -1;
                this.y_boxes(ii) = -1;
            end

            act_AMS = zeros(25,1);
            this.index_AMS = [];

            % checking collisions

            for ii=1:this.n_boxes_tot

                if this.x_boxes(ii)-this.d_boxes(ii)/2<0
                    this.x_boxes(ii) = this.d_boxes(ii)/2;
                elseif this.x_boxes(ii)+this.d_boxes(ii)/2>l_AMS_matrix
                    this.x_boxes(ii) = l_AMS_matrix-this.d_boxes(ii)/2;
                end

                % identifying on which AMS the boxes are (geometrical baricenter)

                if this.y_boxes(ii)<=this.d_AMS
                    this.i_box(ii) = 1;
                elseif this.y_boxes(ii)<=this.d_AMS*2
                    this.i_box(ii) = 2;
                elseif this.y_boxes(ii)<=this.d_AMS*3
                    this.i_box(ii) = 3;
                elseif this.y_boxes(ii)<=this.d_AMS*4
                    this.i_box(ii) = 4;
                elseif this.y_boxes(ii)<=this.d_AMS*5
                    this.i_box(ii) = 5;
                else
                    this.i_box(ii) = 6;
                end

                if this.i_box(ii) == 6
                    this.j_box(ii) = 6;
                elseif this.x_boxes(ii)<=this.d_AMS
                    this.j_box(ii) = 1;
                elseif this.x_boxes(ii)<=this.d_AMS*2
                    this.j_box(ii) = 2;
                elseif this.x_boxes(ii)<=this.d_AMS*3
                    this.j_box(ii) = 3;
                elseif this.x_boxes(ii)<=this.d_AMS*4
                    this.j_box(ii) = 4;
                else
                    this.j_box(ii) = 5;
                end

                % packages kinematics

                if this.i_box(ii) < 6
                    this.x_boxes(ii) = this.x_boxes(ii) + v_AMS(this.i_box(ii),this.j_box(ii))*this.dt*sin(rotation_AMS(this.i_box(ii),this.j_box(ii)));
                    this.y_boxes(ii) = this.y_boxes(ii) + v_AMS(this.i_box(ii),this.j_box(ii))*this.dt*cos(rotation_AMS(this.i_box(ii),this.j_box(ii)));
                    this.vy_boxes(ii) = v_AMS(this.i_box(ii),this.j_box(ii))*cos(rotation_AMS(this.i_box(ii),this.j_box(ii)));
                else
                    this.x_boxes(ii) = this.x_boxes(ii);
                    this.y_boxes(ii) = this.y_boxes(ii) + this.v_treadmill*this.dt;
                    this.vy_boxes(ii) = this.v_treadmill;
                end

                % collisions

                if ii>1 && this.cont>1
                    for jj=ii:-1:2
                        if abs(this.x_boxes(ii)-this.x_boxes(jj-1)) <= this.d_boxes(ii)/2+this.d_boxes(jj-1)/2 + 0.001 ...
                                && abs(this.x_boxes_vect(ii,this.cont-1)-this.x_boxes_vect(jj-1,this.cont-1)) >= this.d_boxes(ii)/2+this.d_boxes(jj-1)/2 ...
                                && abs(this.y_boxes(ii)-this.y_boxes(jj-1)) <= this.d_boxes(ii)/2+this.d_boxes(jj-1)/2 ...
                                && abs(this.y_boxes_vect(ii,this.cont-1)-this.y_boxes_vect(jj-1,this.cont-1)) <= this.d_boxes(ii)/2+this.d_boxes(jj-1)/2
                            if (this.x_boxes(ii)-this.d_boxes(ii)/2 <= this.x_boxes(jj-1)+this.d_boxes(jj-1)/2 && this.x_boxes(ii)-this.d_boxes(ii)/2>this.x_boxes(jj-1)-this.d_boxes(jj-1)/2)
                                penetrazione_x = (this.x_boxes(jj-1)+this.d_boxes(jj-1)/2) - (this.x_boxes(ii)-this.d_boxes(ii)/2);
                                this.x_boxes(ii) = this.x_boxes(ii) + penetrazione_x/2 + this.toll_contatto;
                                this.x_boxes(jj-1) = this.x_boxes(jj-1) - penetrazione_x/2 - this.toll_contatto;
                            elseif (this.x_boxes(ii)+this.d_boxes(ii)/2 >= this.x_boxes(jj-1)-this.d_boxes(jj-1)/2 && this.x_boxes(ii)+this.d_boxes(ii)/2<this.x_boxes(jj-1)+this.d_boxes(jj-1)/2)
                                penetrazione_x = (this.x_boxes(ii)+this.d_boxes(ii)/2) - (this.x_boxes(jj-1)-this.d_boxes(jj-1)/2);
                                this.x_boxes(ii) = this.x_boxes(ii) - penetrazione_x/2 - this.toll_contatto;
                                this.x_boxes(jj-1) = this.x_boxes(jj-1) + penetrazione_x/2 + this.toll_contatto;
                            end
                        elseif abs(this.y_boxes(ii)-this.y_boxes(jj-1)) <= this.d_boxes(ii)/2+this.d_boxes(jj-1)/2 + 0.001 ...
                                && abs(this.y_boxes_vect(ii,this.cont-1)-this.y_boxes_vect(jj-1,this.cont-1)) >= this.d_boxes(ii)/2+this.d_boxes(jj-1)/2 ...
                                && abs(this.x_boxes(ii)-this.x_boxes(jj-1)) <= this.d_boxes(ii)/2+this.d_boxes(jj-1)/2
                            if (this.y_boxes(ii)-this.d_boxes(ii)/2 <= this.y_boxes(jj-1)+this.d_boxes(jj-1)/2 && this.y_boxes(ii)-this.d_boxes(ii)/2>this.y_boxes(jj-1)-this.d_boxes(jj-1)/2) % || (this.y_boxes(jj-1)-this.d_boxes(jj-1)/2 <= this.y_boxes(ii)+this.d_boxes(ii)/2 && this.y_boxes(jj-1)-this.d_boxes(jj-1)/2>this.y_boxes(ii)-this.d_boxes(ii)/2)
                                penetrazione_y = (this.y_boxes(jj-1)+this.d_boxes(jj-1)/2)-(this.y_boxes(ii)-this.d_boxes(ii)/2);
                                this.y_boxes(ii) = this.y_boxes(ii) + penetrazione_y/2 + this.toll_contatto;
                                this.y_boxes(jj-1) = this.y_boxes(jj-1) - penetrazione_y/2 - this.toll_contatto;
                            elseif (this.y_boxes(ii)+this.d_boxes(ii)/2 < this.y_boxes(jj-1)+this.d_boxes(jj-1)/2 && this.y_boxes(ii)+this.d_boxes(ii)/2 >= this.y_boxes(jj-1)-this.d_boxes(jj-1)/2) % || (this.y_boxes(jj-1)+this.d_boxes(jj-1)/2 < this.y_boxes(ii)+this.d_boxes(ii)/2 && this.y_boxes(jj-1)+this.d_boxes(jj-1)/2 >= this.y_boxes(ii)-this.d_boxes(ii)/2)
                                penetrazione_y = (this.y_boxes(ii)+this.d_boxes(ii)/2) - (this.y_boxes(jj-1)-this.d_boxes(jj-1)/2);
                                this.y_boxes(ii) = this.y_boxes(ii) - penetrazione_y/2  - this.toll_contatto;
                                this.y_boxes(jj-1) = this.y_boxes(jj-1) + penetrazione_y/2 + this.toll_contatto;
                            end
                        end
                    end
                end
                
                if this.i_box(ii)<6 && this.j_box(ii)<6
                    act_AMS(this.j_box(ii)+(this.i_box(ii)-1)*5) = 1;
                    this.index_AMS(ii) = this.j_box(ii)+(this.i_box(ii)-1)*5;
                else
                    this.index_AMS(ii) = 0;
                end
            end

            clampBoxPositions(this);
            resolveOverlaps(this);
            clampBoxPositions(this);

            %================= STEP OBSERVATION ==================
           
            Observation = getObservation(this);
            %================= END STEP OBSERVATION ==============

            for ii=1:this.max_gen_boxes
                if ii<=this.n_boxes_tot
                    this.x_boxes_vect(ii,this.cont) = this.x_boxes(ii);
                    this.y_boxes_vect(ii,this.cont) = this.y_boxes(ii);
                    this.x_boxes_prec(ii) = this.x_boxes(ii);
                    this.y_boxes_prec(ii) = this.y_boxes(ii);
                else
                    this.x_boxes_vect(ii,this.cont) = -1;
                    this.y_boxes_vect(ii,this.cont) = -1;
                end
            end

            % Update system states
            this.State = Observation;

            cont_exit = 0;

            % Check terminal condition
            for ii=1:this.n_boxes_tot
                if this.y_boxes(ii)>this.d_AMS*5
                    cont_exit = cont_exit + 1;
                    if this.pack_exited(ii) == 0
                        this.pack_exited(ii) = 1;
                        this.index_exit = this.index_exit + 1;
                        this.exit_order(this.index_exit) = ii;
                        this.exit_time(this.index_exit) = this.time;
                    end
                end
            end

            if cont_exit==this.max_gen_boxes
                IsDone = true;
            else
                IsDone = false;
            end

            this.IsDone = IsDone;
            
            % Get reward
            Reward = getReward(this,AMS_actions);
            
            % (optional) use notifyEnvUpdated to signal that the 
            % environment has been updated (e.g. to update visualization)
            notifyEnvUpdated(this);
        end
        
        % Reset environment to initial state and output initial observation
        % for each episod
        function InitialObservation = reset(this)

            this.index_exit = 0;
            this.LastExitedCount = 0;
            this.pack_exited = zeros(10,1); % is package exited the AMS or not?
            this.exit_order = zeros(10,1); % packages ordered by exit
            this.exit_time = zeros(10,1);

            this.cont_new_box = 1;

            this.n_boxes_tot_vect = 2;
            this.n_boxes_tot = this.n_boxes_tot_vect;

            this.time = 0;

            this.vy_boxes = zeros(this.max_gen_boxes,1);

            this.cont = 0;

            max_d_box = 2.*this.d_AMS; % m
            min_d_box = 0.25*this.d_AMS; % m

            l_AMS_matrix = this.d_AMS*this.n_j_AMS; % m

            % generating initial boxes (2)
            for ii=1:this.max_gen_boxes
                this.d_boxes(ii) = min_d_box + (max_d_box-min_d_box)*rand(1);
            end

            for ii=1:2
                if ii == 1
                    x1 = this.d_boxes(1)*0.5 + (this.d_AMS*2.5-this.d_boxes(1)*0.5)*rand(1);
                else
                    x2 = this.d_AMS*2.5+this.d_boxes(2)*0.5 + (l_AMS_matrix-(this.d_AMS*2.5+this.d_boxes(2)*0.5))*rand(1);
                    if abs(x2-x1) <= this.d_boxes(this.n_boxes_tot)/2+this.d_boxes(this.n_boxes_tot-1)/2
                        x2 = x1 + this.d_boxes(2)/2 + this.d_boxes(1)/2 + 0.1;
                    end
                    if x2-this.d_boxes(2)/2<0
                        x2 = this.d_boxes(2)/2;
                    elseif x2+this.d_boxes(2)/2>l_AMS_matrix
                        x2 = l_AMS_matrix-this.d_boxes(2)/2;
                    end
                    if abs(x2-x1) <= this.d_boxes(2)/2+this.d_boxes(1)/2
                        x1 = x2 - (this.d_boxes(2)/2 + this.d_boxes(1)/2 + 0.1);
                    end
                end

                y1 = 0.001 + rand(1)*0.01;
                y2 = 0.001 + rand(1)*0.05;
            end

            this.x_boxes(1) = x1;
            this.y_boxes(1) = y1;
            this.x_boxes(2) = x2;
            this.y_boxes(2) = y2;

            this.x_boxes_prec(1) = x1;
            this.x_boxes_prec(2) = x2;
            this.y_boxes_prec(1) = y1;
            this.y_boxes_prec(2) = y2;

            for ii=this.n_boxes_tot+1:this.max_gen_boxes
                this.x_boxes(ii) = -1;
                this.y_boxes(ii) = -1;
            end

            act_AMS = zeros(25,1);
            this.index_AMS = zeros(2,1);

            % identifying on which AMS the boxes are (geometrical
            % baricenter)

            for ii=1:this.n_boxes_tot

                if this.y_boxes(ii)<=this.d_AMS
                    this.i_box(ii) = 1;
                elseif this.y_boxes(ii)<=this.d_AMS*2
                    this.i_box(ii) = 2;
                elseif this.y_boxes(ii)<=this.d_AMS*3
                    this.i_box(ii) = 3;
                elseif this.y_boxes(ii)<=this.d_AMS*4
                    this.i_box(ii) = 4;
                elseif this.y_boxes(ii)<=this.d_AMS*5
                    this.i_box(ii) = 5;
                else
                    this.i_box(ii) = 6;
                end

                if this.i_box(ii) == 6
                    this.j_box(ii) = 6;
                elseif this.x_boxes(ii)<=this.d_AMS
                    this.j_box(ii) = 1;
                elseif this.x_boxes(ii)<=this.d_AMS*2
                    this.j_box(ii) = 2;
                elseif this.x_boxes(ii)<=this.d_AMS*3
                    this.j_box(ii) = 3;
                elseif this.x_boxes(ii)<=this.d_AMS*4
                    this.j_box(ii) = 4;
                else
                    this.j_box(ii) = 5;
                end

                if this.i_box(ii)<6 && this.j_box(ii)<6
                    act_AMS(this.j_box(ii)+(this.i_box(ii)-1)*5) = 1;
                    this.index_AMS(ii) = this.j_box(ii)+(this.i_box(ii)-1)*5;
                else
                    this.index_AMS(ii) = 0;
                end

            end

            %================= RESET OBSERVATION =================
           
            InitialObservation = getObservation(this);

            %================= END RESET OBSERVATION =============
            this.State = InitialObservation;
            
            % (optional) use notifyEnvUpdated to signal that the 
            % environment has been updated (e.g. to update visualization)
            notifyEnvUpdated(this);

        end
    end
    %% Optional Methods (set methods' attributes accordingly)
    methods  

        function varargout = plot(this)
            if isempty(this.Visualizer) || ~isvalid(this.Visualizer)
                this.Visualizer = AMSVisualizer_simple(this);
            else
                bringToFront(this.Visualizer);
            end
            if nargout
                varargout{1} = this.Visualizer;
            end
            % Reset Visualizations
            this.VisualizeAnimation = true;
            this.VisualizeActions = false;
            this.VisualizeStates = false;
        end

        % Reward function
        function Reward = getReward(this,AMS_actions)

            %================= REWARD PART =======================
            % main reward terms for singulation
            yCurrent = this.y_boxes(:);
            yPrevious = zeros(this.max_gen_boxes,1);
            if this.cont > 1
                yPrevious = this.y_boxes_vect(1:this.max_gen_boxes,this.cont-1);
            else
                nPrevious = min(numel(this.y_boxes_prec),this.max_gen_boxes);
                if nPrevious > 0
                    yPrevious(1:nPrevious) = this.y_boxes_prec(1:nPrevious);
                end
            end

            forwardProgress = 0;
            for ii = 1:this.n_boxes_tot
                if this.pack_exited(ii) == 0 && yCurrent(ii) > 0
                    forwardProgress = forwardProgress + max(0,yCurrent(ii) - yPrevious(ii));
                end
            end

            exitedBoxes = sum(this.pack_exited);
            newExits = max(0,exitedBoxes - this.LastExitedCount);
            firstNewExit = this.LastExitedCount + 1;
            this.LastExitedCount = exitedBoxes;

            separationReward = 0;
            simultaneousExitPenalty = 0;
            if newExits > 0
                for kk = max(2,firstNewExit):this.index_exit
                    current = this.exit_order(kk);
                    previous = this.exit_order(kk-1);
                    gap = (this.y_boxes(previous) - this.d_boxes(previous)/2) - ...
                        (this.y_boxes(current) + this.d_boxes(current)/2);
                    gapError = gap - this.desired_gap;
                    if gapError >= 0
                        separationReward = separationReward + 45 + 15*tanh(8*gapError);
                    else
                        separationReward = separationReward - 180*abs(gapError);
                    end

                    if abs(this.exit_time(kk)-this.exit_time(kk-1)) < this.min_exit_time_gap
                        simultaneousExitPenalty = simultaneousExitPenalty + 1;
                    end
                end

                if firstNewExit == 1
                    separationReward = separationReward + 8;
                end
            end

            overlapPenalty = 0;
            crowdingPenalty = 0;
            pairGapPenalty = 0;
            pairLeadReward = 0;
            pairSpeedReward = 0;
            pairSpeedPenalty = 0;
            verticalQueuePenalty = 0;
            exitZonePenalty = 0;
            xAlignReward = 0;
            exitLine = this.d_AMS*this.n_i_AMS;
            for ii = 2:this.n_boxes_tot
                for jj = 1:ii-1
                    if this.x_boxes(ii) < 0 || this.x_boxes(jj) < 0
                        continue;
                    end

                    verticalGap = abs(this.y_boxes(ii)-this.y_boxes(jj)) - ...
                        (this.d_boxes(ii)+this.d_boxes(jj))/2;
                    verticalGapError = this.desired_gap - verticalGap;
                    if verticalGapError > 0
                        verticalQueuePenalty = verticalQueuePenalty + ...
                            (verticalGapError/this.desired_gap)^2;
                    end

                    nearExit = this.y_boxes(ii) > exitLine-this.exit_zone_depth || ...
                        this.y_boxes(jj) > exitLine-this.exit_zone_depth;
                    if nearExit && verticalGapError > 0
                        exitZonePenalty = exitZonePenalty + ...
                            1.5*verticalGapError/this.desired_gap;
                    end

                    minLongitudinalGap = (this.d_boxes(ii)+this.d_boxes(jj))/2 + this.desired_gap;
                    actualLongitudinalGap = abs(this.y_boxes(ii)-this.y_boxes(jj));
                    lateralOverlap = (this.d_boxes(ii)+this.d_boxes(jj))/2 - ...
                        abs(this.x_boxes(ii)-this.x_boxes(jj));
                    if lateralOverlap > 0 && actualLongitudinalGap < minLongitudinalGap
                        crowdingPenalty = crowdingPenalty + ...
                            (minLongitudinalGap-actualLongitudinalGap)/this.desired_gap;
                    end

                    xOverlap = (this.d_boxes(ii)+this.d_boxes(jj))/2 - ...
                        abs(this.x_boxes(ii)-this.x_boxes(jj));
                    yOverlap = (this.d_boxes(ii)+this.d_boxes(jj))/2 - ...
                        abs(this.y_boxes(ii)-this.y_boxes(jj));
                    if xOverlap > 0 && yOverlap > 0
                        overlapPenalty = overlapPenalty + xOverlap*yOverlap;
                    end
                end
            end

            centerX = this.d_AMS*this.n_j_AMS/2;
            for ii = 1:this.n_boxes_tot
                if this.x_boxes(ii) > 0
                    localClear = true;
                    for jj = 1:this.n_boxes_tot
                        if ii == jj || this.x_boxes(jj) < 0
                            continue;
                        end
                        edgeGap = abs(this.y_boxes(ii)-this.y_boxes(jj)) - ...
                            (this.d_boxes(ii)+this.d_boxes(jj))/2;
                        if edgeGap < this.desired_gap
                            localClear = false;
                            break;
                        end
                    end
                    if localClear
                        xAlignReward = xAlignReward + ...
                            (1 - abs(this.x_boxes(ii)-centerX)/centerX);
                    end
                end
            end

            for pairStart = 1:2:this.n_boxes_tot-1
                pairNext = pairStart + 1;
                if this.x_boxes(pairStart) < 0 || this.x_boxes(pairNext) < 0
                    continue;
                end

                if this.y_boxes(pairStart) >= this.y_boxes(pairNext)
                    leadBox = pairStart;
                    trailBox = pairNext;
                    leadD = this.d_boxes(pairStart);
                    trailD = this.d_boxes(pairNext);
                    leadY = this.y_boxes(pairStart);
                    trailY = this.y_boxes(pairNext);
                else
                    leadBox = pairNext;
                    trailBox = pairStart;
                    leadD = this.d_boxes(pairNext);
                    trailD = this.d_boxes(pairStart);
                    leadY = this.y_boxes(pairNext);
                    trailY = this.y_boxes(pairStart);
                end
                pairGap = (trailY - trailD/2) - (leadY + leadD/2);
                pairGapError = this.desired_gap - pairGap;
                if pairGapError > 0
                    pairGapPenalty = pairGapPenalty + pairGapError/this.desired_gap;
                else
                    pairLeadReward = pairLeadReward + tanh(4*pairGap);
                end

                speedDiff = this.vy_boxes(leadBox) - this.vy_boxes(trailBox);
                speedScale = this.upperlim_v - this.lowerlim_v;
                if pairGapError > 0
                    if speedDiff > 0
                        pairSpeedReward = pairSpeedReward + speedDiff/speedScale;
                    else
                        pairSpeedPenalty = pairSpeedPenalty + abs(speedDiff)/speedScale;
                    end
                end
            end

            actionEffort = mean(abs(AMS_actions));
            Reward = 1.0*forwardProgress + 3*newExits + separationReward + ...
                2*pairLeadReward + 10*pairSpeedReward + 1*xAlignReward - 40*crowdingPenalty - ...
                80*pairGapPenalty - 30*pairSpeedPenalty - 120*verticalQueuePenalty - ...
                200*exitZonePenalty - 500*overlapPenalty - ...
                300*simultaneousExitPenalty - 1e-3*actionEffort;
            %================= END REWARD PART ===================
                     
        end

        %================= OBSERVATION VECTOR =====================
        % same order as the observation attributes above
        function Observation = getObservation(this)
            boxVelocityY = zeros(this.max_gen_boxes,1);
            nVelocity = min(numel(this.vy_boxes),this.max_gen_boxes);
            if nVelocity > 0
                boxVelocityY(1:nVelocity) = this.vy_boxes(1:nVelocity);
            end

            paddedIndexAMS = zeros(this.max_gen_boxes,1);
            nIndex = min(numel(this.index_AMS),this.max_gen_boxes);
            if nIndex > 0
                paddedIndexAMS(1:nIndex) = this.index_AMS(1:nIndex);
            end

            actionSample = zeros(40,1);
            nAction = min(numel(this.LastAction),numel(actionSample));
            actionSample(1:nAction) = this.LastAction(1:nAction);

            Observation = [
                this.d_boxes(:);
                this.x_boxes(:);
                this.y_boxes(:);
                boxVelocityY(:);
                this.pack_exited(:);
                paddedIndexAMS(:);
                actionSample(:)
            ];
        end
        %================= END OBSERVATION VECTOR =================

        function Metrics = getSingulationMetrics(this)
            exitedOrder = this.exit_order(this.exit_order > 0);
            gaps = zeros(max(0,numel(exitedOrder)-1),1);

            for ii = 2:numel(exitedOrder)
                previous = exitedOrder(ii-1);
                current = exitedOrder(ii);
                gaps(ii-1) = (this.y_boxes(previous) - this.d_boxes(previous)/2) - ...
                    (this.y_boxes(current) + this.d_boxes(current)/2);
            end

            Metrics = struct();
            Metrics.TotalBoxes = this.max_gen_boxes;
            Metrics.ExitedBoxes = sum(this.pack_exited);
            Metrics.ExitedOrder = exitedOrder(:);
            Metrics.ExitTimes = this.exit_time(this.exit_time > 0);
            Metrics.Gaps = gaps;
            Metrics.SuccessfulPairs = sum(gaps >= this.desired_gap);
            Metrics.TotalPairs = numel(gaps);
            Metrics.DesiredGap = this.desired_gap;
            if numel(Metrics.ExitTimes) > 1
                Metrics.ExitTimeGaps = diff(Metrics.ExitTimes);
                Metrics.MinExitTimeGap = min(Metrics.ExitTimeGaps);
                Metrics.SameStepExits = sum(Metrics.ExitTimeGaps < this.min_exit_time_gap);
            else
                Metrics.ExitTimeGaps = [];
                Metrics.MinExitTimeGap = NaN;
                Metrics.SameStepExits = 0;
            end
            if isempty(gaps)
                Metrics.SingulationAccuracy = NaN;
                Metrics.MeanGap = NaN;
                Metrics.MinGap = NaN;
            else
                Metrics.SingulationAccuracy = 100*Metrics.SuccessfulPairs/Metrics.TotalPairs;
                Metrics.MeanGap = mean(gaps);
                Metrics.MinGap = min(gaps);
            end
        end

        function resolveOverlaps(this)
            l_AMS_matrix = this.d_AMS*this.n_j_AMS;

            for pass = 1:2
                for ii = 2:this.n_boxes_tot
                    for jj = 1:ii-1
                        if this.x_boxes(ii) < 0 || this.x_boxes(jj) < 0
                            continue;
                        end

                        xOverlap = (this.d_boxes(ii)+this.d_boxes(jj))/2 - ...
                            abs(this.x_boxes(ii)-this.x_boxes(jj));
                        yOverlap = (this.d_boxes(ii)+this.d_boxes(jj))/2 - ...
                            abs(this.y_boxes(ii)-this.y_boxes(jj));

                        if xOverlap <= 0 || yOverlap <= 0
                            continue;
                        end

                        if xOverlap < yOverlap
                            if this.x_boxes(ii) >= this.x_boxes(jj)
                                direction = 1;
                            else
                                direction = -1;
                            end
                            correction = xOverlap/2 + this.toll_contatto;
                            this.x_boxes(ii) = this.x_boxes(ii) + direction*correction;
                            this.x_boxes(jj) = this.x_boxes(jj) - direction*correction;
                        else
                            if this.y_boxes(ii) >= this.y_boxes(jj)
                                direction = 1;
                            else
                                direction = -1;
                            end
                            correction = yOverlap/2 + this.toll_contatto;
                            this.y_boxes(ii) = this.y_boxes(ii) + direction*correction;
                            this.y_boxes(jj) = this.y_boxes(jj) - direction*correction;
                        end

                        this.x_boxes(ii) = max(this.d_boxes(ii)/2, ...
                            min(l_AMS_matrix-this.d_boxes(ii)/2,this.x_boxes(ii)));
                        this.x_boxes(jj) = max(this.d_boxes(jj)/2, ...
                            min(l_AMS_matrix-this.d_boxes(jj)/2,this.x_boxes(jj)));
                    end
                end
            end
        end

        function clampBoxPositions(this)
            l_AMS_matrix = this.d_AMS*this.n_j_AMS;

            for ii = 1:this.n_boxes_tot
                if this.x_boxes(ii) < 0 || this.y_boxes(ii) < 0
                    continue;
                end

                this.x_boxes(ii) = max(this.d_boxes(ii)/2, ...
                    min(l_AMS_matrix-this.d_boxes(ii)/2,this.x_boxes(ii)));
            end
        end
        
        % (optional) Properties validation through set methods
        function set.State(this,state)
            validateattributes(state,{'numeric'},{'finite','real','vector','numel',100},'','State');
            this.State = double(state(:));
            notifyEnvUpdated(this);
        end
        
    end
    
    methods (Access = protected)
        % (optional) update visualization everytime the environment is updated 
        % (notifyEnvUpdated is called)
        function envUpdatedCallback(this)
        end
    end
end
