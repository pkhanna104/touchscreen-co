#:kivy 1.10.0
#:import Clock kivy.clock.Clock

<Manager>:
    id: screen_manager

    Screen:
        name: 'home'

        BoxLayout
            id: starting
            orientation: 'vertical'

            Label:
                text: ''
                fontsize: 50

            Label:
                text: 'Welcome! Make a Selection in Each Row:'
                fontsize: 80
                color: 0,0,1,1
                canvas.before:
                    Color: 
                        rgb: 1,1,0
                        
            BoxLayout
                id: starting2
                orientation: 'horizontal'
                Label:
                    text: ''
                    fontsize: 50

                Label:
                    text: ''
                    fontsize: 50

                Label:
                    text: 'Animal Names:'
                    color: 1, 1, 0, 1
                    fontsize: 50
                    
                Label: 
                    text: 'Haribo'
                    fontsize: 28
                    halign: 'center'                        
                CheckBox: 
                    group: 'check'
                    id: chk_har
                    active: root.is_haribo  
                     
                Label: 
                    text: 'Fifi'
                    fontsize: 28
                    halign: 'center'                      
                CheckBox: 
                    group: 'check'
                    id: chk_fifi
                    active: root.is_fifi    
                
                Label: 
                    text: 'Nike'
                    fontsize: 28
                    halign: 'center'                 
                CheckBox: 
                    group: 'check'
                    id: chk_nike
                    active: root.is_nike    
                    
                Label: 
                    text: 'Butters'
                    fontsize: 28
                    halign: 'center' 
                CheckBox: 
                    group: 'check'
                    id: chk_but
                    active: root.is_butters    
                    
                Label: 
                    text: 'Testing'
                    fontsize: 28
                    halign: 'center' 
                CheckBox: 
                    group: 'check'
                    id: chk_test
                    active: root.is_testing
                   
                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50

            BoxLayout 
                id: top
                orientation: 'horizontal'
                canvas:
                    Color:
                        rgba: 0.1, 0.1, 0.7, 0.5
                    Rectangle:
                        size: self.size
                        pos: self.pos

                Label:
                    text: ''
                    fontsize: 50

                Label: 
                    text: 'Which Juicer?        '
                    color: 1, 1, 0, 1
           
                Label: 
                    text: 'Old yellow'

                CheckBox:
                    group: 'juicer'
                    id: yellow
                    active: root.juicer_y
            
                Label: 
                    text: 'New red'

                CheckBox:
                    group: 'juicer'
                    id: red
                    active: root.juicer_r

                Label:
                    text: ''
                    fontsize: 50

                Label:
                    text: ''
                    fontsize: 50

            BoxLayout 
                id: top
                orientation: 'horizontal'
                canvas:
                    Color:
                        rgba: 0.1, 0.1, 0.7, 0.5
                    Rectangle:
                        size: self.size
                        pos: self.pos

                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50

                Label: 
                    text: 'Target 1 Timeout'
                    color: 1, 1, 0, 1
                    
                Label: 
                    text: '0.8 sec'
                CheckBox:
                    group: 't1tt'
                    id: t1tt_0pt8_sec
                    active: root.is_t1tt0pt8 
                    
                Label: 
                    text: '1.0 sec'
                CheckBox:
                    group: 't1tt'
                    id: t1tt_1pt0_sec
                    active: root.is_t1tt1pt0  
           
                Label: 
                    text: '1.5 sec'
                CheckBox:
                    group: 't1tt'
                    id: t1tt_1pt5_sec
                    active: root.is_t1tt1pt5    
            
                Label: 
                    text: '2.0 sec'
                CheckBox:
                    group: 't1tt'
                    id: t1tt_2pt0_sec
                    active: root.is_t1tt2pt0
                    
                Label: 
                    text: '2.5 sec'
                CheckBox:
                    group: 't1tt'
                    id: t1tt_2pt5_sec
                    active: root.is_t1tt2pt5

                Label: 
                    text: '3.0 sec'
                CheckBox:
                    group: 't1tt'
                    id: t1tt_3pt0_sec
                    active: root.is_t1tt3pt0
                    
                Label: 
                    text: '3.5 sec'
                CheckBox:
                    group: 't1tt'
                    id: t1tt_3pt5_sec
                    active: root.is_t1tt3pt5
                    
                Label: 
                    text: '4.0 sec'
                CheckBox:
                    group: 't1tt'
                    id: t1tt_4pt0_sec
                    active: root.is_t1tt4pt0
            
                Label: 
                    text: '10.0 sec'
                CheckBox: 
                    group: 't1tt'
                    id: t1tt_10pt0_sec
                    active: root.is_t1tt10pt0

                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50
            
            BoxLayout 
                id: top
                orientation: 'horizontal'
                canvas:
                    Color:
                        rgba: 0.1, 0.1, 0.7, 0.5
                    Rectangle:
                        size: self.size
                        pos: self.pos

                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50

                Label: 
                    text: 'Target 2+ Timeout'
                    color: 1, 1, 0, 1
           
           
                Label: 
                    text: '0.7 sec'
                CheckBox:
                    group: 'tt'
                    id: tt_0pt7_sec
                    active: root.is_tt0pt7  
                    
                Label: 
                    text: '0.8 sec'
                CheckBox:
                    group: 'tt'
                    id: tt_0pt8_sec
                    active: root.is_tt0pt8  
                    
                Label: 
                    text: '0.9 sec'
                CheckBox:
                    group: 'tt'
                    id: tt_0pt9_sec
                    active: root.is_tt0pt9  
                    
                Label: 
                    text: '1.0 sec'
                CheckBox:
                    group: 'tt'
                    id: tt_1pt0_sec
                    active: root.is_tt1pt0
                    
                Label: 
                    text: '1.1 sec'
                CheckBox:
                    group: 'tt'
                    id: tt_1pt1_sec
                    active: root.is_tt1pt1
                    
                Label: 
                    text: '1.2 sec'
                CheckBox:
                    group: 'tt'
                    id: tt_1pt2_sec
                    active: root.is_tt1pt2
                    
                Label: 
                    text: '1.3 sec'
                CheckBox:
                    group: 'tt'
                    id: tt_1pt3_sec
                    active: root.is_tt1pt3
                
                Label: 
                    text: '1.5 sec'
                CheckBox:
                    group: 'tt'
                    id: tt_1pt5_sec
                    active: root.is_tt1pt5    
            
                Label: 
                    text: '2.0 sec'
                CheckBox:
                    group: 'tt'
                    id: tt_2pt0_sec
                    active: root.is_tt2pt0
                    
                Label: 
                    text: '2.5 sec'
                CheckBox:
                    group: 'tt'
                    id: tt_2pt5_sec
                    active: root.is_tt2pt5

                Label: 
                    text: '3.0 sec'
                CheckBox:
                    group: 'tt'
                    id: tt_3pt0_sec
                    active: root.is_tt3pt0
                    
                Label: 
                    text: '3.5 sec'
                CheckBox:
                    group: 'tt'
                    id: tt_3pt5_sec
                    active: root.is_tt3pt5
                    
                Label: 
                    text: '4.0 sec'
                CheckBox:
                    group: 'tt'
                    id: tt_4pt0_sec
                    active: root.is_tt4pt0
            
                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50
                    
            BoxLayout
                id: drag
                orientation: 'horizontal'
                Label:
                    text: ''
                    fontsize: 50

                Label:
                    text: ''
                    fontsize: 50

                Label:
                    text: 'Drag OK:'
                    color: 1, 1, 0, 1
                    fontsize: 50
                    
                Label: 
                    text: 'YES'                     
                CheckBox: 
                    group: 'drag'
                    id: dragok
                    active: root.is_dragok  
                     
                Label: 
                    text: 'NO'                   
                CheckBox: 
                    group: 'drag'
                    id: dragnotok
                    active: root.is_dragnotok    
                   
                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50
            
            
            BoxLayout:
                id: params:
                orientation: 'horizontal'
                        
                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50


                Label: 
                    text: 'Crashbar Hold Time'
                    color: 1, 1, 0, 1
                    
                Label:
                    text: ''
                    fontsize: 50

                Label: 
                    text: 'NO Crashbar'
                CheckBox: 
                    id: bhtfalse
                    group: 'button_hold_time'
                    active: root.is_bhtfalse

                Label: 
                    text: '0.0 sec'
                CheckBox: 
                    id: bht000
                    group: 'button_hold_time'
                    active: root.is_bht000

                Label:  
                    text: '0.1 sec'
                CheckBox:
                    id: bht100
                    group: 'button_hold_time'
                    active: root.is_bht100

                Label:  
                    text: '0.2 sec'
                CheckBox:
                    id: bht200
                    group: 'button_hold_time'
                    active: root.is_bht200

                Label:  
                    text: '0.3 sec'
                CheckBox:
                    id: bht300
                    group: 'button_hold_time'
                    active: root.is_bht300

                Label: 
                    text: '0.4 sec'
                CheckBox:
                    id: bht400
                    group: 'button_hold_time'
                    active: root.is_bht400

                Label: 
                    text: '0.5 sec'
                CheckBox:
                    id: bht500
                    group: 'button_hold_time'
                    active: root.is_bht500

                Label: 
                    text: '0.6 sec'
                CheckBox:
                    id: bht600
                    group: 'button_hold_time'
                    active: root.is_bht600

                Label:
                    text: '0.7 sec'
                CheckBox:
                    id: bht700
                    group: 'button_hold_time'
                    active: root.is_bht700
                    
                Label:
                    text: '0.8 sec'
                CheckBox:
                    id: bht800
                    group: 'button_hold_time'
                    active: root.is_bht800
                    
                Label:
                    text: '0.9 sec'
                CheckBox:
                    id: bht900
                    group: 'button_hold_time'
                    active: root.is_bht900
                    
                Label:
                    text: '1.0 sec'
                CheckBox:
                    id: bht1000
                    group: 'button_hold_time'
                    active: root.is_bht1000

                Label: 
                    text: '.2-.4s'
                CheckBox:
                    id: bht200to400
                    group: 'button_hold_time'
                    active: root.is_bht200to400
                    
                Label:
                    text: '0.6-0.8 sec'
                CheckBox:
                    id: bht600to800
                    group: 'button_hold_time'
                    active: root.is_bht600to800
                    
                Label:
                    text: '0.8-1.0 sec'
                CheckBox:
                    id: bht800to1000
                    group: 'button_hold_time'
                    active: root.is_bht800to1000

                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50
                    
                    
                    
                    
            BoxLayout 
                id: button_rew
                orientation: 'horizontal'
                canvas:
                    Color:
                        rgba: 0.2, 0.2, 0.5, 0.5
                    Rectangle:
                        size: self.size
                        pos: self.pos

                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50

                Label: 
                    text: 'Crashbar Reward'
                    color: 1, 1, 0, 1

                Label:
                    text: '0.0 sec'
                CheckBox:
                    id: button_rew_zero_sec
                    group: 'button_rew'
                    active: root.is_bhrew000

                Label: 
                    text: '0.1 sec'
                CheckBox:
                    id: button_rew_pt1_sec
                    group: 'button_rew'
                    active: root.is_bhrew100

                Label: 
                    text: '0.3 sec'
                CheckBox:
                    id: button_rew_pt3_sec
                    group: 'button_rew'
                    active: root.is_bhrew300

                Label: 
                    text: '0.5 sec'
                CheckBox:
                    id: button_rew_pt5_sec
                    group: 'button_rew'
                    active: root.is_bhrew500

                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50




            BoxLayout:
                id: params:
                orientation: 'horizontal'

                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50

                Label: 
                    text: 'Target Hold Time'
                    color: 1, 1, 0, 1

                Label: 
                    text: '0.0 sec'
                CheckBox:
                    id: tht000
                    group: 'hold_time'
                    active: root.is_tht000

                Label: 
                    text: '0.1 sec'
                CheckBox: 
                    id: tht100
                    group: 'hold_time'
                    active: root.is_tht100

                Label: 
                    text: '0.2 sec'
                CheckBox: 
                    id: tht200
                    group: 'hold_time'
                    active: root.is_tht200

                Label: 
                    text: '0.3 sec'
                CheckBox: 
                    id: tht300
                    group: 'hold_time'
                    active: root.is_tht300

                Label: 
                    text: '0.4 sec'
                CheckBox:
                    id: tht400
                    group: 'hold_time'
                    active: root.is_tht400

                Label: 
                    text: '0.5 sec'
                CheckBox:
                    id: tht500
                    group: 'hold_time'
                    active: root.is_tht500

                Label: 
                    text: '0.6 sec'
                CheckBox:
                    id: tht600
                    group: 'hold_time'
                    active: root.is_tht600
                    
                Label:
                    text: '0.1-0.3'
                CheckBox:
                    id: tht100to300
                    group: 'hold_time'
                    active: root.is_tht100to300

                Label:
                    text: '0.4-0.6'
                CheckBox:
                    id: tht400to600
                    group: 'hold_time'
                    active: root.is_tht400to600

                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50

            BoxLayout 
                id: min_rew
                orientation: 'horizontal'
                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50
                Label: 
                    text: 'Minimum Reward'
                    color: 1, 1, 0, 1

                Label: 
                    text: 'No Reward Scaling'
                CheckBox:
                    id: min_rew_none
                    group: 'min_rew'
                    active: root.is_minthrewnone
                
                Label: 
                    text: '0.0 sec'
                CheckBox:
                    id: min_rew_zero_sec
                    group: 'min_rew'
                    active: root.is_minthrew000

                Label: 
                    text: '0.1 sec'
                CheckBox:
                    id: min_rew_pt1_sec
                    group: 'min_rew'
                    active: root.is_minthrew100

                Label: 
                    text: '0.2 sec'
                CheckBox:
                    id: min_rew_pt2_sec
                    group: 'min_rew'
                    active: root.is_minthrew200

                Label: 
                    text: '0.3 sec'
                CheckBox:
                    id: min_rew_pt3_sec
                    group: 'min_rew'
                    active: root.is_minthrew300
                    
                Label: 
                    text: '0.4 sec'
                CheckBox:
                    id: min_rew_pt4_sec
                    group: 'min_rew'
                    active: root.is_minthrew400
                    
                Label: 
                    text: '0.5 sec'
                CheckBox:
                    id: min_rew_pt5_sec
                    group: 'min_rew'
                    active: root.is_minthrew500
                    
                Label: 
                    text: '0.6 sec'
                CheckBox:
                    id: min_rew_pt6_sec
                    group: 'min_rew'
                    active: root.is_minthrew600
                    
                Label: 
                    text: '0.7 sec'
                CheckBox:
                    id: min_rew_pt7_sec
                    group: 'min_rew'
                    active: root.is_minthrew700

                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50
            
            BoxLayout 
                id: big_rew
                orientation: 'horizontal'
                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50
                Label: 
                    text: 'Maximum Reward'
                    color: 1, 1, 0, 1

                Label: 
                    text: '0.3 sec'
                CheckBox:
                    id: big_rew_pt3_sec
                    group: 'big_rew'
                    active: root.is_threw300

                Label: 
                    text: '0.5 sec'
                CheckBox:
                    id: big_rew_pt5_sec
                    group: 'big_rew'
                    active: root.is_threw500

                Label: 
                    text: '0.7 sec'
                CheckBox:
                    id: big_rew_pt7_sec
                    group: 'big_rew'
                    active: root.is_threw700
                    
                Label: 
                    text: '0.9 sec'
                CheckBox:
                    id: big_rew_pt9_sec
                    group: 'big_rew'
                    active: root.is_threw900
                    
                Label: 
                    text: '1.1 sec'
                CheckBox:
                    id: big_rew_1pt1_sec
                    group: 'big_rew'
                    active: root.is_threw1100

                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50
            
            # BoxLayout 
            #     id: big_rew
            #     orientation: 'horizontal'
            #     canvas:
            #         Color:
            #             rgba: 0.1, 0.1, 0.7, 0.5
            #         Rectangle:
            #             size: self.size
            #             pos: self.pos
            #     Label:
            #         text: ''
            #         fontsize: 50
            #     Label:
            #         text: ''
            #         fontsize: 50
                    
            #     Label: 
            #         text: 'Reward Variability'
            #         color: 1, 1, 0, 1

            #     Label: 
            #         text: 'Reward All'
            #     CheckBox:
            #         id: rew_all
            #         group: 'rew_var'
            #         active: root.is_rewvarall
                    
            #     Label: 
            #         text: 'Reward 50% '
            #     CheckBox:
            #         id: rew_50
            #         group: 'rew_var'
            #         active: root.is_rewvar50

            #     Label: 
            #         text: 'Reward 33% w/ 10% doubled'
            #     CheckBox:
            #         id: rew_30
            #         group: 'rew_var'
            #         active: root.is_rewvar33


            #     Label:
            #         text: ''
            #         fontsize: 50
            #     Label:
            #         text: ''
            #         fontsize: 50





            BoxLayout:
                id: params:
                orientation: 'horizontal'

                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50

                Label: 
                    text: 'Appearing Target Radius'
                    color: 1, 1, 0, 1
                
                Label:
                    text: ''
                    fontsize: 50

                Label: 
                    text: '.5 cm'
                CheckBox:
                    id: targ_rad_5
                    group: 'targ_cm'
                    active: root.is_trad050

                Label: 
                    text: '.75 cm'
                CheckBox:
                    id: targ_rad_75
                    group: 'targ_cm'
                    active: root.is_trad075
                    
                Label: 
                    text: '.82 cm'
                CheckBox:
                    id: targ_rad_82
                    group: 'targ_cm'
                    active: root.is_trad082

                Label: 
                    text: '.91 cm'
                CheckBox:
                    id: targ_rad_91
                    group: 'targ_cm'
                    active: root.is_trad091

                Label: 
                    text: '1.0 cm'
                CheckBox:
                    id: targ_rad_10
                    group: 'targ_cm'
                    active: root.is_trad100
                    
                Label: 
                    text: '1.5 cm'
                CheckBox:
                    id: targ_rad_15
                    group: 'targ_cm'
                    active: root.is_trad150
                    
                Label: 
                    text: '1.85cm'
                CheckBox:
                    id: targ_rad_18
                    group: 'targ_cm'
                    active: root.is_trad185

                Label: 
                    text: '2.25cm'
                CheckBox:
                    id: targ_rad_22
                    group: 'targ_cm'
                    active: root.is_trad225

                Label:
                    text: '3.0 cm'
                CheckBox:
                    id: targ_rad_30
                    group: 'targ_cm'
                    active: root.is_trad300
                    
                Label:
                    text: '4.0 cm'
                CheckBox:
                    id: targ_rad_40
                    group: 'targ_cm'
                    active: root.is_trad400
                                        
                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50
                    
            BoxLayout:
                id: params:
                orientation: 'horizontal'

                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50

                Label: 
                    text: 'Effective Target Radius'
                    color: 1, 1, 0, 1
                
                Label:
                    text: ''
                    fontsize: 50

                Label: 
                    text: 'Same as Appears'
                CheckBox:
                    id: eff_targ_rad_same
                    group: 'eff_targ_cm'
                    active: root.is_efftradsame

                Label: 
                    text: '1.0 cm'
                CheckBox:
                    id: eff_targ_rad_10
                    group: 'eff_targ_cm'
                    active: root.is_efftrad10
                    
                Label: 
                    text: '2.0 cm'
                CheckBox:
                    id: eff_targ_rad_20
                    group: 'eff_targ_cm'
                    active: root.is_efftrad20
                    
                Label: 
                    text: '3.0 cm'
                CheckBox:
                    id: eff_targ_rad_30
                    group: 'eff_targ_cm'
                    active: root.is_efftrad30
                    
                Label: 
                    text: '4.0 cm'
                CheckBox:
                    id: eff_targ_rad_40
                    group: 'eff_targ_cm'
                    active: root.is_efftrad40
                    
                Label: 
                    text: '5.0 cm'
                CheckBox:
                    id: eff_targ_rad_50
                    group: 'eff_targ_cm'
                    active: root.is_efftrad50
                                        
                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50
                    
            
            BoxLayout:
                id: outlines
                orientation: 'horizontal'
                canvas:
                    Color:
                        rgba: 0.1, 0.1, 0.7, 0.5
                    Rectangle:
                        size: self.size
                        pos: self.pos

                Label: 
                    text: 'Show Outlines'
                    color: 1, 1, 0, 1
                
                Label: 
                    text: 'Yes'
                CheckBox:
                    id: outlines_true
                    group: 'outlines'
                    active: root.is_outlines
                
                Label: 
                    text: 'No'
                CheckBox:
                    id: outlines_false
                    group: 'outlines'
                    active: root.is_no_outlines
            
            
            BoxLayout:
                id: seq
                orientation: 'horizontal'
                canvas:
                    Color:
                        rgba: 0.1, 0.1, 0.7, 0.5
                    Rectangle:
                        size: self.size
                        pos: self.pos

                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50

                Label: 
                    text: 'Sequence'
                    color: 1, 1, 0, 1
                
#                Label: 
#                    text: 'A'
#                CheckBox:
#                    id: seqA
#                    group: 'seq'
#                    active: root.is_seqA
#                
#                Label: 
#                    text: 'B'
#                CheckBox:
#                    id: seqB
#                    group: 'seq'
#                    active: root.is_seqB
#                    
#                Label: 
#                    text: 'C'
#                CheckBox:
#                    id: seqC
#                    group: 'seq'
#                    active: root.is_seqC
#                    
#                Label: 
#                    text: 'D'
#                CheckBox:
#                    id: seqD
#                    group: 'seq'
#                    active: root.is_seqD
#                    
#                Label: 
#                    text: 'E'
#                CheckBox:
#                    id: seqE
#                    group: 'seq'
#                    active: root.is_seqE
#                    
#                Label: 
#                    text: 'F'
#                CheckBox:
#                    id: seqF
#                    group: 'seq'
#                    active: root.is_seqF
#                    
#                Label: 
#                    text: 'G'
#                CheckBox:
#                    id: seqG
#                    group: 'seq'
#                    active: root.is_seqG
#                
#                Label: 
#                    text: 'H'
#                CheckBox:
#                    id: seqH
#                    group: 'seq'
#                    active: root.is_seqH
#                    
#                Label: 
#                    text: 'I'
#                CheckBox:
#                    id: seqI
#                    group: 'seq'
#                    active: root.is_seqI
#                    
#                Label: 
#                    text: 'J'
#                CheckBox:
#                    id: seqJ
#                    group: 'seq'
#                    active: root.is_seqJ
#                    
#                Label: 
#                    text: 'K'
#                CheckBox:
#                    id: seqK
#                    group: 'seq'
#                    active: root.is_seqK
#                    
#               Label: 
#                    text: 'L'
#                CheckBox:
#                    id: seqL
#                    group: 'seq'
#                    active: root.is_seqL
#                    
#                Label: 
#                    text: 'M'
#                CheckBox:
#                    id: seqM
#                    group: 'seq'
#                    active: root.is_seqM
#                    
#                Label: 
#                    text: 'N'
#                CheckBox:
#                    id: seqN
#                    group: 'seq'
#                    active: root.is_seqN
#                    
#                Label: 
#                    text: 'O'
#                CheckBox:
#                    id: seqO
#                    group: 'seq'
#                    active: root.is_seqO
#                    
#                Label: 
#                    text: 'P'
#                CheckBox:
#                    id: seqP
#                    group: 'seq'
#                    active: root.is_seqP
#                    
#                Label: 
#                    text: 'Q'
#                CheckBox:
#                    id: seqQ
#                    group: 'seq'
#                    active: root.is_seqQ
#                    
#                Label: 
#                    text: 'R'
#                CheckBox:
#                    id: seqR
#                    group: 'seq'
#                    active: root.is_seqR
#                    
#                Label: 
#                    text: 'S'
#                CheckBox:
#                    id: seqS
#                    group: 'seq'
#                    active: root.is_seqS
#                    
#                Label: 
#                    text: 'T'
#                CheckBox:
#                    id: seqT
#                    group: 'seq'
#                    active: root.is_seqT
#                   
#                Label: 
#                    text: 'U'
#                CheckBox:
#                    id: seqU
#                    group: 'seq'
#                    active: root.is_seqU
#                   
#                Label: 
#                    text: 'V'
#                CheckBox:
#                    id: seqV
#                    group: 'seq'
#                    active: root.is_seqV   
#                    
#                Label: 
#                    text: 'W'
#                CheckBox:
#                    id: seqW
#                    group: 'seq'
#                    active: root.is_seqW    
#                    
#                Label: 
#                    text: 'X'
#                CheckBox:
#                    id: seqX
#                    group: 'seq'
#                    active: root.is_seqX
                    
                Label: 
                    text: 'Y'
                CheckBox:
                    id: seqY
                    group: 'seq'
                    active: root.is_seqY
                    
                Label: 
                    text: 'Random 5'
                CheckBox:
                    id: seqRand5
                    group: 'seq'
                    active: root.is_seqRand5   
                    
                Label: 
                    text: 'Repeat Last'
                CheckBox:
                    id: seqRepeat
                    group: 'seq'
                    active: root.is_seqRepeat 
                    
                Label: 
                    text: 'Random Every'
                CheckBox:
                    id: seqRandomEvery
                    group: 'seq'
                    active: root.is_seqRandomEvery
                    
                Label: 
                    text: 'Cen Out'
                CheckBox:
                    id: centerOut
                    group: 'seq'
                    active: root.is_CO
                    
                Label: 
                    text: 'But Out'
                CheckBox: 
                    id: buttonOut
                    group: 'seq'
                    active: root.is_BO

                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50
                    
                    
                    
            BoxLayout:
                id: t1_pos
                orientation: 'horizontal'
                canvas:
                    Color:
                        rgba: 0.1, 0.1, 0.7, 0.5
                    Rectangle:
                        size: self.size
                        pos: self.pos

                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50

                Label: 
                    text: 'Target 1 Position'
                    color: 1, 1, 0, 1
                
                Label: 
                    text: 'Random'
                CheckBox:
                    id: t1random
                    group: 't1_pos'
                    active: root.is_t1rand

                Label: 
                    text: 'Center'
                CheckBox:
                    id: t1center
                    group: 't1_pos'
                    active: root.is_t1cent
                
                Label: 
                    text: 'Upper Left'
                CheckBox:
                    id: t1upper_left
                    group: 't1_pos'
                    active: root.is_t1ul
                    
                Label: 
                    text: 'Middle Left'
                CheckBox:
                    id: t1mid_left
                    group: 't1_pos'
                    active: root.is_t1ml

                Label: 
                    text: 'Lower Left'
                CheckBox:
                    id: t1lower_left
                    group: 't1_pos'
                    active: root.is_t1ll
                    
                Label: 
                    text: 'Upper Middle'
                CheckBox:
                    id: t1upper_mid
                    group: 't1_pos'
                    active: root.is_t1um

                Label: 
                    text: 'Lower Middle'
                CheckBox:
                    id: t1lower_mid
                    group: 't1_pos'
                    active: root.is_t1lm
                    
                Label: 
                    text: 'Upper Right'
                CheckBox:
                    id: t1upper_right
                    group: 't1_pos'
                    active: root.is_t1ur
                    
                Label: 
                    text: 'Middle Right'
                CheckBox:
                    id: t1mid_right
                    group: 't1_pos'
                    active: root.is_t1mr

                Label: 
                    text: 'Lower Right'
                CheckBox:
                    id: t1lower_right
                    group: 't1_pos'
                    active: root.is_t1lr
                                        
                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50
            
            
            
            
            # BoxLayout:
            #     id: nudge_x_t1
            #     orientation: 'horizontal'

                # Label:
                #     text: ''
                #     fontsize: 50
                # Label:
                #     text: ''
                #     fontsize: 50

                # Label: 
                #     text: 'Target 1 Nudge X'
                #     color: 1, 1, 0, 1

                # Label: 
                #     text: '-6 cm'
                # CheckBox:
                #     id: nudge_x_t1_neg6
                #     group: 'nudge_x_t1'
                #     active: root.is_t1nudgeneg6
                    
                # Label: 
                #     text: '-4 cm'
                # CheckBox:
                #     id: nudge_x_t1_neg4
                #     group: 'nudge_x_t1'
                #     active: root.is_t1nudgeneg4

                # Label: 
                #     text: '-2 cm'
                # CheckBox:
                #     id: nudge_x_t1_neg2
                #     group: 'nudge_x_t1'
                #     active: root.is_t1nudgeneg2

                # Label: 
                #     text: '0 cm'
                # CheckBox:
                #     id: nudge_x_t1_zero
                #     group: 'nudge_x_t1'
                #     active: root.is_t1nudgezero

                # Label: 
                #     text: '+2 cm'
                # CheckBox:
                #     id: nudge_x_t1_pos2
                #     group: 'nudge_x_t1'
                #     active: root.is_t1nudgepos2
                    
                # Label: 
                #     text: '+4 cm'
                # CheckBox:
                #     id: nudge_x_t1_pos4
                #     group: 'nudge_x_t1'
                #     active: root.is_t1nudgepos4

                # Label: 
                #     text: '+6 cm'
                # CheckBox:
                #     id: nudge_x_t1_pos6
                #     active: root.is_t1nudgepos6

                # Label:
                #     text: ''
                #     fontsize: 50
                # Label:
                #     text: ''
                #     fontsize: 50

                    
            BoxLayout:
                id: t2_pos
                orientation: 'horizontal'
                canvas:
                    Color:
                        rgba: 0.1, 0.1, 0.7, 0.5
                    Rectangle:
                        size: self.size
                        pos: self.pos

                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50

                Label: 
                    text: 'Target 2 Position'
                    color: 1, 1, 0, 1
                
                Label: 
                    text: 'None'
                CheckBox:
                    id: t2none
                    group: 't2_pos'
                    active: root.is_t2none

                Label: 
                    text: 'Random'
                CheckBox:
                    id: t2random
                    group: 't2_pos'
                    active: root.is_t2rand
                
                Label: 
                    text: 'Center'
                CheckBox:
                    id: t2center
                    group: 't2_pos'
                    active: root.is_t2cent
                
                Label: 
                    text: 'Upper Left'
                CheckBox:
                    id: t2upper_left
                    group: 't2_pos'
                    active: root.is_t2ul
                    
                Label: 
                    text: 'Middle Left'
                CheckBox:
                    id: t2mid_left
                    group: 't2_pos'
                    active: root.is_t2ml

                Label: 
                    text: 'Lower Left'
                CheckBox:
                    id: t2lower_left
                    group: 't2_pos'
                    active: root.is_t2ll
                    
                Label: 
                    text: 'Upper Middle'
                CheckBox:
                    id: t2upper_mid
                    group: 't2_pos'
                    active: root.is_t2um

                Label: 
                    text: 'Lower Middle'
                CheckBox:
                    id: t2lower_mid
                    group: 't2_pos'
                    active: root.is_t2lm
                    
                Label: 
                    text: 'Upper Right'
                CheckBox:
                    id: t2upper_right
                    group: 't2_pos'
                    active: root.is_t2ur
                    
                Label: 
                    text: 'Middle Right'
                CheckBox:
                    id: t2mid_right
                    group: 't2_pos'
                    active: root.is_t2mr

                Label: 
                    text: 'Lower Right'
                CheckBox:
                    id: t2lower_right
                    group: 't2_pos'
                    active: root.is_t2lr


                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50
                    
                    
            # BoxLayout:
            #     id: nudge_x_t2
            #     orientation: 'horizontal'

            #     Label:
            #         text: ''
            #         fontsize: 50
            #     Label:
            #         text: ''
            #         fontsize: 50

            #     Label: 
            #         text: 'Target 2 Nudge X'
            #         color: 1, 1, 0, 1

            #     Label: 
            #         text: '-6 cm'
            #     CheckBox:
            #         id: nudge_x_t2_neg6
            #         group: 'nudge_x_t2'
            #         active: root.is_t2nudgeneg6
                    
            #     Label: 
            #         text: '-4 cm'
            #     CheckBox:
            #         id: nudge_x_t2_neg4
            #         group: 'nudge_x_t2'
            #         active: root.is_t2nudgeneg4

            #     Label: 
            #         text: '-2 cm'
            #     CheckBox:
            #         id: nudge_x_t2_neg2
            #         group: 'nudge_x_t2'
            #         active: root.is_t2nudgeneg2

            #     Label: 
            #         text: '0 cm'
            #     CheckBox:
            #         id: nudge_x_t2_zero
            #         group: 'nudge_x_t2'
            #         active: root.is_t2nudgezero

            #     Label: 
            #         text: '+2 cm'
            #     CheckBox:
            #         id: nudge_x_t2_pos2
            #         group: 'nudge_x_t2'
            #         active: root.is_t2nudgepos2
                    
            #     Label: 
            #         text: '+4 cm'
            #     CheckBox:
            #         id: nudge_x_t2_pos4
            #         group: 'nudge_x_t2'
            #         active: root.is_t2nudgepos4

            #     Label: 
            #         text: '+6 cm'
            #     CheckBox:
            #         id: nudge_x_t2_pos6
            #         group: 'nudge_x_t2'
            #         active: root.is_t2nudgepos6

            #     Label:
            #         text: ''
            #         fontsize: 50
            #     Label:
            #         text: ''
            #         fontsize: 50
            
            
            BoxLayout:
                id: t3_pos
                orientation: 'horizontal'
                canvas:
                    Color:
                        rgba: 0.1, 0.1, 0.7, 0.5
                    Rectangle:
                        size: self.size
                        pos: self.pos

                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50

                Label: 
                    text: 'Target 3 Position'
                    color: 1, 1, 0, 1
                    
                Label: 
                    text: 'None'
                CheckBox:
                    id: t3none
                    group: 't3_pos'
                    active: root.is_t3none
                
                Label: 
                    text: 'Center'
                CheckBox:
                    id: t3center
                    group: 't3_pos'
                    active: root.is_t3cent
                
                Label: 
                    text: 'Upper Left'
                CheckBox:
                    id: t3upper_left
                    group: 't3_pos'
                    active: root.is_t3ul
                    
                Label: 
                    text: 'Middle Left'
                CheckBox:
                    id: t3mid_left
                    group: 't3_pos'
                    active: root.is_t3ml

                Label: 
                    text: 'Lower Left'
                CheckBox:
                    id: t3lower_left
                    group: 't3_pos'
                    active: root.is_t3ll
                    
                Label: 
                    text: 'Upper Middle'
                CheckBox:
                    id: t3upper_mid
                    group: 't3_pos'
                    active: root.is_t3um

                Label: 
                    text: 'Lower Middle'
                CheckBox:
                    id: t3lower_mid
                    group: 't3_pos'
                    active: root.is_t3lm
                    
                Label: 
                    text: 'Upper Right'
                CheckBox:
                    id: t3upper_right
                    group: 't3_pos'
                    active: root.is_t3ur
                    
                Label: 
                    text: 'Middle Right'
                CheckBox:
                    id: t3mid_right
                    group: 't3_pos'
                    active: root.is_t3mr

                Label: 
                    text: 'Lower Right'
                CheckBox:
                    id: t3lower_right
                    group: 't3_pos'
                    active: root.is_t3lr


                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50
             
            
            # BoxLayout:
            #     id: nudge_x_t3
            #     orientation: 'horizontal'

            #     Label:
            #         text: ''
            #         fontsize: 50
            #     Label:
            #         text: ''
            #         fontsize: 50

            #     Label: 
            #         text: 'Target 3 Nudge X'
            #         color: 1, 1, 0, 1

            #     Label: 
            #         text: '-6 cm'
            #     CheckBox:
            #         id: nudge_x_t3_neg6
            #         group: 'nudge_x_t3'
            #         active: root.is_t3nudgeneg6
                    
            #     Label: 
            #         text: '-4 cm'
            #     CheckBox:
            #         id: nudge_x_t3_neg4
            #         group: 'nudge_x_t3'
            #         active: root.is_t3nudgeneg4

            #     Label: 
            #         text: '-2 cm'
            #     CheckBox:
            #         id: nudge_x_t3_neg2
            #         group: 'nudge_x_t3'
            #         active: root.is_t3nudgeneg2

            #     Label: 
            #         text: '0 cm'
            #     CheckBox:
            #         id: nudge_x_t3_zero
            #         group: 'nudge_x_t3'
            #         active: root.is_t3nudgezero

            #     Label: 
            #         text: '+2 cm'
            #     CheckBox:
            #         id: nudge_x_t3_pos2
            #         group: 'nudge_x_t3'
            #         active: root.is_t3nudgepos2
                    
            #     Label: 
            #         text: '+4 cm'
            #     CheckBox:
            #         id: nudge_x_t3_pos4
            #         group: 'nudge_x_t3'
            #         active: root.is_t3nudgepos4

            #     Label: 
            #         text: '+6 cm'
            #     CheckBox:
            #         id: nudge_x_t3_pos6
            #         group: 'nudge_x_t3'
            #         active: root.is_t3nudgepos6

            #     Label:
            #         text: ''
            #         fontsize: 50
            #     Label:
            #         text: ''
            #         fontsize: 50
    
                    
            BoxLayout:
                id: t4_pos
                orientation: 'horizontal'
                canvas:
                    Color:
                        rgba: 0.1, 0.1, 0.7, 0.5
                    Rectangle:
                        size: self.size
                        pos: self.pos

                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50

                Label: 
                    text: 'Target 4 Position'
                    color: 1, 1, 0, 1
                
                Label: 
                    text: 'None'
                CheckBox:
                    id: t4none
                    group: 't4_pos'
                    active: root.is_t4none
                
                Label: 
                    text: 'Center'
                CheckBox:
                    id: t4center
                    group: 't4_pos'
                    active: root.is_t4cent
                
                Label: 
                    text: 'Upper Left'
                CheckBox:
                    id: t4upper_left
                    group: 't4_pos'
                    active: root.is_t4ul
                    
                Label: 
                    text: 'Middle Left'
                CheckBox:
                    id: t4mid_left
                    group: 't4_pos'
                    active: root.is_t4ml

                Label: 
                    text: 'Lower Left'
                CheckBox:
                    id: t4lower_left
                    group: 't4_pos'
                    active: root.is_t4ll
                    
                Label: 
                    text: 'Upper Middle'
                CheckBox:
                    id: t4upper_mid
                    group: 't4_pos'
                    active: root.is_t4um

                Label: 
                    text: 'Lower Middle'
                CheckBox:
                    id: t4lower_mid
                    group: 't4_pos'
                    active: root.is_t4lm
                    
                Label: 
                    text: 'Upper Right'
                CheckBox:
                    id: t4upper_right
                    group: 't4_pos'
                    active: root.is_t4ur
                    
                Label: 
                    text: 'Middle Right'
                CheckBox:
                    id: t4mid_right
                    group: 't4_pos'
                    active: root.is_t4mr

                Label: 
                    text: 'Lower Right'
                CheckBox:
                    id: t4lower_right
                    group: 't4_pos'
                    active: root.is_t4lr

                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50
                    
                    
            # BoxLayout:
            #     id: nudge_x_t4
            #     orientation: 'horizontal'

            #     Label:
            #         text: ''
            #         fontsize: 50
            #     Label:
            #         text: ''
            #         fontsize: 50

            #     Label: 
            #         text: 'Target 4 Nudge X'
            #         color: 1, 1, 0, 1

            #     Label: 
            #         text: '-6 cm'
            #     CheckBox:
            #         id: nudge_x_t4_neg6
            #         group: 'nudge_x_t4'
            #         active: root.is_t4nudgeneg6
                    
            #     Label: 
            #         text: '-4 cm'
            #     CheckBox:
            #         id: nudge_x_t4_neg4
            #         group: 'nudge_x_t4'
            #         active: root.is_t4nudgeneg4

            #     Label: 
            #         text: '-2 cm'
            #     CheckBox:
            #         id: nudge_x_t4_neg2
            #         group: 'nudge_x_t4'
            #         active: root.is_t4nudgeneg2

            #     Label: 
            #         text: '0 cm'
            #     CheckBox:
            #         id: nudge_x_t4_zero
            #         group: 'nudge_x_t4'
            #         active: root.is_t4nudgezero

            #     Label: 
            #         text: '+2 cm'
            #     CheckBox:
            #         id: nudge_x_t4_pos2
            #         group: 'nudge_x_t4'
            #         active: root.is_t4nudgepos2
                    
            #     Label: 
            #         text: '+4 cm'
            #     CheckBox:
            #         id: nudge_x_t4_pos4
            #         group: 'nudge_x_t4'
            #         active: root.is_t4nudgepos4

            #     Label: 
            #         text: '+6 cm'
            #     CheckBox:
            #         id: nudge_x_t4_pos6
            #         group: 'nudge_x_t4'
            #         active: root.is_t4nudgepos6

            #     Label:
            #         text: ''
            #         fontsize: 50
            #     Label:
            #         text: ''
            #         fontsize: 50
                      
            
            BoxLayout:
                id: t5_pos
                orientation: 'horizontal'
                canvas:
                    Color:
                        rgba: 0.1, 0.1, 0.7, 0.5
                    Rectangle:
                        size: self.size
                        pos: self.pos

                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50

                Label: 
                    text: 'Target 5 Position'
                    color: 1, 1, 0, 1
                
                Label: 
                    text: 'None'
                CheckBox:
                    id: t5none
                    group: 't5_pos'
                    active: root.is_t5none
                
                Label: 
                    text: 'Center'
                CheckBox:
                    id: t5center
                    group: 't5_pos'
                    active: root.is_t5cent
                
                Label: 
                    text: 'Upper Left'
                CheckBox:
                    id: t5upper_left
                    group: 't5_pos'
                    active: root.is_t5ul
                    
                Label: 
                    text: 'Middle Left'
                CheckBox:
                    id: t5mid_left
                    group: 't5_pos'
                    active: root.is_t5ml

                Label: 
                    text: 'Lower Left'
                CheckBox:
                    id: t5lower_left
                    group: 't5_pos'
                    active: root.is_t5ll
                    
                Label: 
                    text: 'Upper Middle'
                CheckBox:
                    id: t5upper_mid
                    group: 't5_pos'
                    active: root.is_t5um

                Label: 
                    text: 'Lower Middle'
                CheckBox:
                    id: t5lower_mid
                    group: 't5_pos'
                    active: root.is_t5lm
                    
                Label: 
                    text: 'Upper Right'
                CheckBox:
                    id: t5upper_right
                    group: 't5_pos'
                    active: root.is_t5ur
                    
                Label: 
                    text: 'Middle Right'
                CheckBox:
                    id: t5mid_right
                    group: 't5_pos'
                    active: root.is_t5mr

                Label: 
                    text: 'Lower Right'
                CheckBox:
                    id: t5lower_right
                    group: 't5_pos'
                    active: root.is_t5lr

                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50
            
            
            
            BoxLayout:
                id: screen_top
                orientation: 'horizontal'
                canvas:
                    Color:
                        rgba: 0.1, 0.1, 0.7, 0.5
                    Rectangle:
                        size: self.size
                        pos: self.pos

                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50

                Label: 
                    text: 'Lower Screen Top By'
                    color: 1, 1, 0, 1

                Label: 
                    text: '0 cm'
                CheckBox:
                    id: screen_top_zero
                    group: 'screen_top'
                    active: root.is_screentopzero
                    
                Label: 
                    text: '2 cm'
                CheckBox:
                    id: screen_top_neg1
                    group: 'screen_top'
                    active: root.is_screentop2
                
                Label: 
                    text: '4 cm'
                CheckBox:
                    id: screen_top_neg2
                    group: 'screen_top'
                    active: root.is_screentop4
                    
                Label: 
                    text: '6 cm'
                CheckBox:
                    id: screen_top_neg3
                    group: 'screen_top'
                    active: root.is_screentop6
                
                Label: 
                    text: '8 cm'
                CheckBox:
                    id: screen_top_neg4
                    group: 'screen_top'
                    active: root.is_screentop8
                  
                Label: 
                    text: '10 cm'
                CheckBox:
                    id: screen_top_neg5
                    group: 'screen_top'
                    active: root.is_screentop10
                    
                Label: 
                    text: '12 cm'
                CheckBox:
                    id: screen_top_neg6
                    group: 'screen_top'
                    active: root.is_screentop12

                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50
                    
            BoxLayout:
                id: screen_bot
                orientation: 'horizontal'
                canvas:
                    Color:
                        rgba: 0.1, 0.1, 0.7, 0.5
                    Rectangle:
                        size: self.size
                        pos: self.pos

                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50

                Label: 
                    text: 'Raise Screen Bottom By'
                    color: 1, 1, 0, 1

                Label: 
                    text: '0 cm'
                CheckBox:
                    id: screen_bot_0
                    group: 'screen_bot'
                    active: root.is_screenbot0
                    
                Label: 
                    text: '2 cm'
                CheckBox:
                    id: screen_bot_2
                    group: 'screen_bot'
                    active: root.is_screenbot2
                
                Label: 
                    text: '4 cm'
                CheckBox:
                    id: screen_bot_4
                    group: 'screen_bot'
                    active: root.is_screenbot4
                    
                Label: 
                    text: '6 cm'
                CheckBox:
                    id: screen_bot_6
                    group: 'screen_bot'
                    active: root.is_screenbot6
                
                Label: 
                    text: '8 cm'
                CheckBox:
                    id: screen_bot_8
                    group: 'screen_bot'
                    active: root.is_screenbot8
                  
                Label: 
                    text: '10 cm'
                CheckBox:
                    id: screen_bot_10
                    group: 'screen_bot'
                    active: root.is_screenbot10
                    
                Label: 
                    text: '12 cm'
                CheckBox:
                    id: screen_bot_12
                    group: 'screen_bot'
                    active: root.is_screenbot12

                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50
            
            
            BoxLayout:
                id: time2next_targ
                orientation: 'horizontal'

                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50

                Label: 
                    text: 'Time til next targ appears'
                    color: 1, 1, 0, 1

                Label: 
                    text: 'Never'
                CheckBox:
                    id: t2next_never
                    group: 'time2next_targ'
                    active: root.is_ttntnever
                    
                Label: 
                    text: '0.25 sec'
                CheckBox:
                    id: t2next_250
                    group: 'time2next_targ'
                    active: root.is_ttnt025
                
                Label: 
                    text: '0.5 sec'
                CheckBox:
                    id: t2next_500
                    group: 'time2next_targ'
                    active: root.is_ttnt050
                    
                Label: 
                    text: '0.75'
                CheckBox:
                    id: t2next_750
                    group: 'time2next_targ'
                    active: root.is_ttnt075
                
                Label: 
                    text: '1.0 sec'
                CheckBox:
                    id: t2next_1000
                    group: 'time2next_targ'
                    active: root.is_ttnt100
                  
                Label: 
                    text: '1.5 sec'
                CheckBox:
                    id: t2next_1500
                    group: 'time2next_targ'
                    active: root.is_ttnt150

                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50
                    
            BoxLayout:
                id: intertarg_delay
                orientation: 'horizontal'

                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50

                Label: 
                    text: 'Intertarget delay'
                    color: 1, 1, 0, 1

                Label: 
                    text: '0 sec'
                CheckBox:
                    id: intertarg_delay0
                    group: 'intertarg_delay'
                    active: root.is_inttargdelay0
                    
                Label: 
                    text: '0.1 sec'
                CheckBox:
                    id: intertarg_delay100
                    group: 'intertarg_delay'
                    active: root.is_inttargdelay100
                
                Label: 
                    text: '0.15 sec'
                CheckBox:
                    id: intertarg_delay150
                    group: 'intertarg_delay'
                    active: root.is_inttargdelay150
                    
                Label: 
                    text: '0.2 sec'
                CheckBox:
                    id: intertarg_delay200
                    group: 'intertarg_delay'
                    active: root.is_inttargdelay200
                
                Label: 
                    text: '0.25 sec'
                CheckBox:
                    id: intertarg_delay250
                    group: 'intertarg_delay'
                    active: root.is_inttargdelay250
                    
                Label: 
                    text: '0.3 sec'
                CheckBox:
                    id: intertarg_delay300
                    group: 'intertarg_delay'
                    active: root.is_inttargdelay300
                    
                Label: 
                    text: '0.4 sec'
                CheckBox:
                    id: intertarg_delay400
                    group: 'intertarg_delay'
                    active: root.is_inttargdelay400
                    
                Label: 
                    text: '0.5 sec'
                CheckBox:
                    id: intertarg_delay500
                    group: 'intertarg_delay'
                    active: root.is_inttargdelay500

                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50
            
            
            
            BoxLayout:
                id: break
                orientation: 'horizontal'
                canvas:
                    Color:
                        rgba: 0.1, 0.1, 0.7, 0.5
                    Rectangle:
                        size: self.size
                        pos: self.pos
                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50

                Label: 
                    text: 'Break Every: '
                    color: 1, 1, 0, 1

                Label: 
                    text: 'NO Break'
                CheckBox:
                    id: break_none
                    group: 'break'
                    active: root.is_nobreak
                
                Label: 
                    text: '10 trials'
                CheckBox:
                    id: break_10
                    group: 'break'
                    active: root.is_break10

                Label: 
                    text: '15 trials'
                CheckBox:
                    id: break_15
                    group: 'break'
                    active: root.is_break15

                Label:
                    text: '20 trials'
                CheckBox:
                    id: break_20
                    group: 'break'
                    active: root.is_break20
                    
                Label:
                    text: '25 trials'
                CheckBox:
                    id: break_25
                    group: 'break'
                    active: root.is_break25

                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50        
            
            
            BoxLayout:
                id: breakdur
                orientation: 'horizontal'
                canvas:
                    Color:
                        rgba: 0.1, 0.1, 0.7, 0.5
                    Rectangle:
                        size: self.size
                        pos: self.pos
                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50

                Label: 
                    text: 'Break Length: '
                    color: 1, 1, 0, 1

                Label: 
                    text: '0.5 min'
                CheckBox:
                    id: breakdur30
                    group: 'breakdur'
                    active: root.is_breakdur30
                
                Label: 
                    text: '1 min'
                CheckBox:
                    id: breakdur60
                    group: 'breakdur'
                    active: root.is_breakdur60

                Label: 
                    text: '1.5 min'
                CheckBox:
                    id: breakdur90
                    group: 'breakdur'
                    active: root.is_breakdur90

                Label:
                    text: '2 min'
                CheckBox:
                    id: breakdur120
                    group: 'breakdur'
                    active: root.is_breakdur120
                    
                Label:
                    text: '2.5 min'
                CheckBox:
                    id: breakdur150
                    group: 'breakdur'
                    active: root.is_breakdur150

                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50         
            
            BoxLayout:
                id: test
                orientation: 'horizontal'
                canvas:
                    Color:
                        rgba: 0.1, 0.1, 0.7, 0.5
                    Rectangle:
                        size: self.size
                        pos: self.pos
                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50

                Label: 
                    text: 'Auto-Quit After: '
                    color: 1, 1, 0, 1

                Label: 
                    text: '10 trials'
                CheckBox:
                    id: ten_trials
                    group: 'trials'
                    active: root.is_autoqt10

                Label: 
                    text: '25 trials'
                CheckBox:
                    id: twenty_five_trials
                    group: 'trials'
                    active: root.is_autoqt25

                Label:
                    text: '50 trials'
                CheckBox:
                    id: fifty_trials
                    group: 'trials'
                    active: root.is_autoqt50
                    
                Label:
                    text: '60 trials'
                CheckBox:
                    id: sixty_trials
                    group: 'trials'
                    active: root.is_autoqt60
                    
                Label:
                    text: '90 trials'
                CheckBox:
                    id: ninety_trials
                    group: 'trials'
                    active: root.is_autoqt90
                    
                Label:
                    text: '100 trials!'
                CheckBox:
                    id: hundred_trials
                    group: 'trials'
                    active: root.is_autoqt100

                Label:
                    text: 'Dont AutoQuit'
                CheckBox:
                    id: no_trials
                    group: 'trials'
                    active: root.is_autoqtnever

                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50
                Label:
                    text: ''
                    fontsize: 50


            Button:
                text: 'Play Target Chase Game'
                halign: 'center'
                valign: 'middle'
                font_size: 50
                text_size: self.size
                on_release: 
                    root.current = 'splash_start'
                    splash.init(dict(haribo=chk_har.active, fifi=chk_fifi.active, nike=chk_nike.active, butters=chk_but.active, testing=chk_test.active), 
                    dict(button_rew=[button_rew_zero_sec.active, button_rew_pt1_sec.active, button_rew_pt3_sec.active, button_rew_pt5_sec.active], 
                    min_rew=[min_rew_none.active, min_rew_zero_sec.active, min_rew_pt1_sec.active, min_rew_pt2_sec.active, min_rew_pt3_sec.active, min_rew_pt4_sec.active, min_rew_pt5_sec.active, min_rew_pt6_sec.active, min_rew_pt7_sec.active],
                    big_rew=[big_rew_pt3_sec.active, big_rew_pt5_sec.active, big_rew_pt7_sec.active, big_rew_pt9_sec.active, big_rew_1pt1_sec.active]), 
                    dict(targ_rad=[targ_rad_5.active, targ_rad_75.active, targ_rad_82.active, targ_rad_91.active, targ_rad_10.active, targ_rad_15.active, targ_rad_18.active, targ_rad_22.active, targ_rad_30.active, targ_rad_40.active], 
                    eff_targ_rad=[eff_targ_rad_same.active, eff_targ_rad_10.active, eff_targ_rad_20.active, eff_targ_rad_30.active, eff_targ_rad_40.active, eff_targ_rad_50.active],
#                    seq=[seqA.active, seqB.active, seqC.active, seqD.active, seqE.active, seqF.active, seqG.active, seqH.active, seqI.active, seqJ.active, seqK.active, seqL.active, seqM.active, seqN.active, seqO.active, seqP.active, seqQ.active, seqR.active, seqS.active, seqT.active, seqU.active, seqV.active, seqW.active, centerOut.active, buttonOut.active],
                    seq=[seqY.active, seqRand5.active, seqRepeat.active, seqRandomEvery.active, centerOut.active, buttonOut.active],
                    targ1_pos=[t1random.active, t1center.active, t1upper_left.active, t1mid_left.active, t1lower_left.active, t1upper_mid.active, t1lower_mid.active, t1upper_right.active, t1mid_right.active, t1lower_right.active],
                    targ2_pos=[t2none.active, t2random.active, t2center.active, t2upper_left.active, t2mid_left.active, t2lower_left.active, t2upper_mid.active, t2lower_mid.active, t2upper_right.active, t2mid_right.active, t2lower_right.active],
                    targ3_pos=[t4none.active, t3center.active, t3upper_left.active, t3mid_left.active, t3lower_left.active, t3upper_mid.active, t3lower_mid.active, t3upper_right.active, t3mid_right.active, t3lower_right.active],
                    targ4_pos=[t4none.active, t4center.active, t4upper_left.active, t4mid_left.active, t4lower_left.active, t4upper_mid.active, t4lower_mid.active, t4upper_right.active, t4mid_right.active, t4lower_right.active],
                    targ5_pos=[t5none.active, t5center.active, t5upper_left.active, t5mid_left.active, t5lower_left.active, t5upper_mid.active, t5lower_mid.active, t5upper_right.active, t5mid_right.active, t5lower_right.active],
                    time_to_next_targ=[t2next_never.active, t2next_250.active, t2next_500.active, t2next_750.active, t2next_1000.active, t2next_1500.active],
                    intertarg_delay=[intertarg_delay0.active, intertarg_delay100.active, intertarg_delay150.active, intertarg_delay200.active, intertarg_delay250.active, intertarg_delay300.active, intertarg_delay400.active, intertarg_delay500.active],
                    outlines=[outlines_true.active, outlines_false.active]),
                    dict(button_hold=[bhtfalse.active, bht000.active, bht100.active, bht200.active, bht300.active, bht400.active, bht500.active, bht600.active, bht700.active, bht800.active, bht900.active, bht1000.active, bht200to400.active, bht600to800.active, bht800to1000.active],
                    hold=[tht000.active, tht100.active, tht200.active, tht300.active, tht400.active, tht500.active, tht600.active, tht100to300.active, tht400to600.active]), 
                    dict(autoquit=[ten_trials.active, twenty_five_trials.active, fifty_trials.active, sixty_trials.active, ninety_trials.active, hundred_trials.active, no_trials.active]),
                    # dict(rew_var=[rew_all.active, rew_50.active, rew_30.active]),
                    dict(t1tt=[t1tt_0pt8_sec.active, t1tt_1pt0_sec.active, t1tt_1pt5_sec.active, t1tt_2pt0_sec.active, t1tt_2pt5_sec.active, t1tt_3pt0_sec.active, t1tt_3pt5_sec.active, t1tt_4pt0_sec.active, t1tt_10pt0_sec.active],
                    tt=[tt_0pt7_sec.active, tt_0pt8_sec.active, tt_0pt9_sec.active, tt_1pt0_sec.active, tt_1pt1_sec.active, tt_1pt2_sec.active, tt_1pt3_sec.active, tt_1pt5_sec.active, tt_2pt0_sec.active, tt_2pt5_sec.active, tt_3pt0_sec.active, tt_3pt5_sec.active, tt_4pt0_sec.active]),
                    dict(drag=[dragok.active, dragnotok.active]),
                    # dict(nudge_x_t1=[nudge_x_t1_neg6.active, nudge_x_t1_neg4.active, nudge_x_t1_neg2.active, nudge_x_t1_zero.active, nudge_x_t1_pos2.active, nudge_x_t1_pos4.active, nudge_x_t1_pos6.active],
                    # nudge_x_t2=[nudge_x_t2_neg6.active, nudge_x_t2_neg4.active, nudge_x_t2_neg2.active, nudge_x_t2_zero.active, nudge_x_t2_pos2.active, nudge_x_t2_pos4.active, nudge_x_t2_pos6.active],
                    # nudge_x_t3=[nudge_x_t3_neg6.active, nudge_x_t3_neg4.active, nudge_x_t3_neg2.active, nudge_x_t3_zero.active, nudge_x_t3_pos2.active, nudge_x_t3_pos4.active, nudge_x_t3_pos6.active],
                    # nudge_x_t4=[nudge_x_t4_neg6.active, nudge_x_t4_neg4.active, nudge_x_t4_neg2.active, nudge_x_t4_zero.active, nudge_x_t4_pos2.active, nudge_x_t4_pos4.active, nudge_x_t4_pos6.active]),
                    dict(screen_top=[screen_top_neg6.active, screen_top_neg5.active, screen_top_neg4.active, screen_top_neg3.active, screen_top_neg2.active, screen_top_neg1.active, screen_top_zero.active], 
                    screen_bot=[screen_bot_0.active, screen_bot_2.active, screen_bot_4.active, screen_bot_6.active, screen_bot_8.active, screen_bot_10.active, screen_bot_12.active]),
                    dict(juicer=[yellow.active, red.active]),
                    dict(breaktrl=[break_none.active, break_10.active, break_15.active, break_20.active, break_25.active],
                    breakdur=[breakdur30.active, breakdur60.active, breakdur90.active, breakdur120.active, breakdur150.active]))

            Label:
                text: ''
                fontsize: 50
            Label:
                text: ''
                fontsize: 50

    Screen: 
        name: 'splash_start'
        Splash:
            id: splash

        BoxLayout:
            id: test
            orientation: 'vertical'
         
            Label:
                text: 'Parameters Cached!! \n \n Prepare the Monkey! \n \n Pausing Until Start Button Pressed. '
                halign: 'center'
                fontsize: 50

            Label:
                text: ''
                fontsize: 50

            Label:
                text: ''
                fontsize: 50

            Label:
                text: ''
                fontsize: 50

            Label:
                text: ''
                fontsize: 50
                
            BoxLayout:
                id: test2
                orientation: 'horizontal'

                Label:
                    text: ''
                    fontsize: 50
                
                Label:
                    text: ''
                    fontsize: 50

                Label:
                    text: ''
                    fontsize: 50

                Label:
                    text: ''
                    fontsize: 50

                Button: 
                    text: 'Start'
                    color: .25, .25, .25, 1
                    background_color: (.2, .2, .2, 1.0)

                    on_release:
                        root.current = 'game_screen'
                        game.init(*splash.args)
                        Clock.schedule_interval(game.update, 1.0 / 60.0)
                
                Label:
                    text: ''
                    fontsize: 50

                Label:
                    text: ''
                    fontsize: 50

            Label:
                text: ''
                fontsize: 50
                
    Screen:
        name: 'game_screen'
        COGame:
            id: game

<COGame>:
    target1: target_1
    target2: target_2
    target1_in: target_one_in
    target1_out: target_one_out
    target2_in: target_two_in
    target2_out: target_two_out
    target3_in: target_three_in
    target3_out: target_three_out
    target4_in: target_four_in
    target4_out: target_four_out
    target5_in: target_five_in
    target5_out: target_five_out
    target6_in: target_six_in
    target6_out: target_six_out
    target7_in: target_seven_in
    target7_out: target_seven_out
    target8_in: target_eight_in
    target8_out: target_eight_out
    target9_in: target_nine_in
    target9_out: target_nine_out
    exit_target1: target_e1
    exit_target2: target_e2
    pd1_indicator_targ: pd1_indicator
    pd2_indicator_targ: pd2_indicator
    vid_indicator_targ: vid_indicator

    Target:
        id: pd1_indicator
        
    Target:
        id: pd2_indicator
        
    Target:
        id: vid_indicator
        
    Target:
        id: target_one_out
    
    Target:
        id: target_one_in
        
    Target:
        id: target_two_out
    
    Target:
        id: target_two_in
        
    Target:
        id: target_three_out
    
    Target:
        id: target_three_in
        
    Target:
        id: target_four_out
    
    Target:
        id: target_four_in
        
    Target:
        id: target_five_out
    
    Target:
        id: target_five_in
        
    Target:
        id: target_six_out
    
    Target:
        id: target_six_in
        
    Target:
        id: target_seven_out
    
    Target:
        id: target_seven_in
        
    Target:
        id: target_eight_out
    
    Target:
        id: target_eight_in
        
    Target:
        id: target_nine_out
    
    Target:
        id: target_nine_in
        
    Target: 
        id: target_1
        
    Target: 
        id: target_2

    Target:
        id: target_e1

    Target:
        id: target_e2

    Label:
        color: .5,.5,.5,1
        font_size: 50
        center_x: root.width * 15/16
        top: root.height * 1/16
        text: str(root.trial_counter)

    
    Label:
        color: .5,.5,.5,1
        font_size: 50
        center_x: root.width * 2/8
        top: root.top - 170
        text: root.percent_correct_text
    Label:
        color: .5,.5,.5,1
        font_size: 50
        center_x: root.width * 5/8
        top: root.top - 170
        text: root.percent_correct

    Label:
        color: .5,.5,.5,1
        font_size: 50
        center_x: root.width * 2/8
        top: root.top - 270
        text: root.tht_text
    Label:
        color: .5,.5,.5,1
        font_size: 50
        center_x: root.width * 5/8
        top: root.top - 270
        text: root.tht_param


    Label:
        color: .5,.5,.5,1
        font_size: 50
        center_x: root.width * 2/8
        top: root.top - 470
        text: root.targ_size_text
    Label:
        color: .5,.5,.5,1
        font_size: 50
        center_x: root.width * 5/8
        top: root.top - 470
        text: root.targ_size_param


    Label:
        color: .5,.5,.5,1
        font_size: 50
        center_x: root.width * 2/8
        top: root.top - 570
        text: root.big_rew_text
    Label:
        color: .5,.5,.5,1
        font_size: 50
        center_x: root.width * 5/8
        top: root.top - 570
        text: root.big_rew_time_param


<Target>:
    canvas:
        Color:
            rgba:self.color
        Ellipse: 
            pos:self.pos
            size:self.size