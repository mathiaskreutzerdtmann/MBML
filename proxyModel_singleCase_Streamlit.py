# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 12:39:27 2025

@author: mkreutzerdtman
"""
import streamlit as st
 
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid
import sys
import matplotlib.patches as patches
import matplotlib.animation as animation
import tempfile
from matplotlib.animation import FFMpegWriter
import time

year_in_seconds = 31.536e6 #31.536e6s is one year
mD_to_m2 = 9.869233e-16 #1mD = 9.869233e-16m2


def progress_bar(iteration, total, length=40):
    percent = (iteration / total) * 100
    bar = "#" * int(length * iteration / total) + "-" * (length - int(length * iteration / total))
    sys.stdout.write(f"\r[{bar}] {percent:.2f}%")
    sys.stdout.flush()
    
def draw_analitic(seaLevel,lastBarrierDepth,injectionBase,plumeSize_h,plumeSize_v,pressureFront_h,pressureFront_v,presSeal,sealLimit,presInj,injLimit,ax):

    ax.cla()
    scale = 4e-5
    verticalExagg = 5

    # Add a rectangle
    rect_sea = patches.Rectangle(
        (0, 1.0-seaLevel*scale*verticalExagg), 1.0, seaLevel*scale*verticalExagg,  # Bottom-left corner (x, y), width, height
        edgecolor='lightblue', facecolor='lightblue', linewidth=2, alpha=0.8
    )
    ax.add_patch(rect_sea)
    rect_lastSeal = patches.Rectangle(
        (0, 1.0-lastBarrierDepth*scale*verticalExagg-seaLevel*scale*verticalExagg), 1.0, lastBarrierDepth*scale*verticalExagg,  # Bottom-left corner (x, y), width, height
        edgecolor='lightgreen', facecolor='lightgreen', linewidth=2, alpha=0.8
    )
    ax.add_patch(rect_lastSeal)
    rect_sand = patches.Rectangle(
        (0, 0), 1.0, 1.0-lastBarrierDepth*scale*verticalExagg-seaLevel*scale*verticalExagg,  # Bottom-left corner (x, y), width, height
        edgecolor='yellow', facecolor='yellow', linewidth=2, alpha=0.2
    )
    ax.add_patch(rect_sand)

    
    
    ellipse = patches.Rectangle(
    (0.5-pressureFront_h*scale/2,  1.0-injectionBase*scale*verticalExagg-pressureFront_v*scale*verticalExagg/2-plumeSize_v*scale*verticalExagg/2), pressureFront_h*scale, pressureFront_v*scale*verticalExagg,
    edgecolor='red', facecolor='pink', linewidth=1, alpha=0.1
)
    ellipse2 = patches.Rectangle(
    (0.5-plumeSize_h*scale/2, 1.0-injectionBase*scale*verticalExagg-plumeSize_v*scale*verticalExagg), plumeSize_h*scale, plumeSize_v*scale*verticalExagg, 
    edgecolor='darkgreen', facecolor='green', linewidth=1, alpha=0.1
)

    img_pil = Image.open("pressureFront.png")
        
    # Resize the image (force aspect ratio change)
    if(pressureFront_h>0):
        img_resized = img_pil.resize((100, int(100*pressureFront_v/pressureFront_h*verticalExagg)))
    
        # Convert back to NumPy array for Matplotlib
        img_array = np.array(img_resized)
        
        # Display the image stretched to fit the rectangle
        ax.imshow(img_array, extent=[
            0.5 - pressureFront_h * scale / 2,  # Left X
            0.5 + pressureFront_h * scale / 2,  # Right X
            1.0 - injectionBase * scale * verticalExagg - pressureFront_v * scale * verticalExagg/2-plumeSize_v*scale*verticalExagg/2,  # Bottom Y
            1.0 - injectionBase * scale * verticalExagg + pressureFront_v * scale * verticalExagg/2-plumeSize_v*scale*verticalExagg/2 # Top Y
        ], aspect='auto')  # 'auto' allows stretching

    img_pil2 = Image.open("plume.png")
        
    # Resize the image (force aspect ratio change)
    if(plumeSize_h>0):
        img_resized2 = img_pil2.resize((100, int(100*plumeSize_v/plumeSize_h*verticalExagg)))
    
        # Convert back to NumPy array for Matplotlib
        img_array2 = np.array(img_resized2)
        
        # Display the image stretched to fit the rectangle
        ax.imshow(img_array2, extent=[
            0.5 - plumeSize_h * scale / 2,  # Left X
            0.5 + plumeSize_h * scale / 2,  # Right X
            1.0 - injectionBase * scale * verticalExagg -plumeSize_v*scale*verticalExagg,  # Bottom Y
            1.0 - injectionBase * scale * verticalExagg  # Top Y
        ], aspect='auto')  # 'auto' allows stretching



    ax.add_patch(ellipse)
    ax.add_patch(ellipse2)
    if(presSeal>sealLimit):
        ax.text(0.01, 1.0-0.9*lastBarrierDepth*scale*verticalExagg-seaLevel*scale*verticalExagg, f"Pressure just below seal [geomec. limit] in bars: {presSeal/1e5:.1f} [{sealLimit/1e5:.1f}]", fontsize=12, color='red', bbox=dict(facecolor='white', alpha=0.5))
    else:
        ax.text(0.01, 1.0-0.9*lastBarrierDepth*scale*verticalExagg-seaLevel*scale*verticalExagg, f"Pressure just below seal [geomec. limit] in bars: {presSeal/1e5:.1f} [{sealLimit/1e5:.1f}]", fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.5))
    if(presInj>injLimit):
        ax.text(0.01, 1.0-1.3*injectionBase*scale*verticalExagg-plumeSize_v*scale*verticalExagg, f"Pressure on injection region [geomec. limit] in bars: {presInj/1e5:.1f} [{injLimit/1e5:.1f}]", fontsize=12, color='red', bbox=dict(facecolor='white', alpha=0.5))
    else:
        ax.text(0.01, 1.0-1.3*injectionBase*scale*verticalExagg-plumeSize_v*scale*verticalExagg, f"Pressure on injection region [geomec. limit] in bars: {presInj/1e5:.1f} [{injLimit/1e5:.1f}]", fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.5))

    # Set limits and remove axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    
# generate animation
def generate_simulation_animation(CO2_equivalentRadius,CO2_plumeHeight,H_spread_pressure,V_spread_pressure,pressure_on_bottom_last_formation_barrier,geomecGradient_shallow,percentage_geomecLimits_shallow,overpressure_coreArea_percent,percentage_geomecLimits_deeper,t,dotm_i,m_i,capacity_constrained):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax3 = ax2.twinx()

    def update(i):
        ax1.clear()
        ax2.clear()
        ax3.clear()



        draw_analitic(
            SeaWaterLevel,
            bottom_last_formation_barrier,
            injection_base,
            CO2_equivalentRadius[i],
            CO2_plumeHeight[i],
            H_spread_pressure[i],
            V_spread_pressure[i],
            pressure_on_bottom_last_formation_barrier[i] + bottom_last_formation_barrier * Pressure_gradient,
            (geomecGradient_shallow * percentage_geomecLimits_shallow) * bottom_last_formation_barrier,
            (1 + overpressure_coreArea_percent[i]) * (injection_base * Pressure_gradient),
            (geomecGradient_deeper * percentage_geomecLimits_deeper) * injection_base,
            ax1
        )

        ax2.plot(t[:i+1]/year_in_seconds, dotm_i[:i+1]*year_in_seconds/1e9, color='tab:red', label='Injection rate [million t/y]')
        ax2.set_xlim([0, max(t)/year_in_seconds])
        ax2.set_ylim([0, max(dotm_i*year_in_seconds/1e9)*1.1])
        ax2.set_xlabel("Time [y]")
        ax2.set_ylabel("Injection rate")
        ax2.legend(loc='upper left')
        ax2.grid(True)
        
        # Second y-axis: another variable
        ax3.plot(t[:i+1]/year_in_seconds, m_i[:i+1]/1e9, color='tab:blue', label='Injected mass [million tons]')
        # Find the first index where m_i >= C
        threshold_index = next((j for j, val in enumerate(m_i[:i+1]) if val >= capacity_constrained), None)
        
        # Add a star marker at that point
        if threshold_index is not None:
            ax3.plot(t[threshold_index]/year_in_seconds, m_i[threshold_index]/1e9,
                     marker='*', markersize=12, color='black', label='Geomec. limit reached')
        ax3.set_ylim([0, max(m_i/1e9)*1.1])
        ax3.set_ylabel("Injected mass")
        ax3.legend(loc='upper right')
        
        plt.tight_layout()
        return fig,

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=range(0, len(t), 5),  # step = 5 for speed
        interval=200,                # 200 ms between frames
        blit=False
    )

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
 
        writer = FFMpegWriter(fps=10, bitrate=1800)
        ani.save(tmpfile.name, writer=writer)
        tmpfile.seek(0)
        buf = tmpfile.name
    return buf

def proxy_model_CCS(Datum,bottom_last_formation_barrier,SeaWaterLevel,phi,k,kvkh_ratio,LCC,C_Land,Solubility,Pressure_gradient,geomecGradient_shallow,geomecGradient_deeper,Pwf,well_count,injection_base,injection_interval,maximumRate,maximumAllowedPlumeReach,minimumAllowedLevel,percentage_geomecLimits_shallow,percentage_geomecLimits_deeper,squaredArea):
    
    t = np.linspace(0, 40*year_in_seconds,100)

    ## secondary inputs (function of inputs)
    
    NTG = 0.6+(0.7-LCC) 
    NTG = min(NTG,0.9)
    
    Ks = 100
    Kr = 0.3
    
    ##pre adjust 0.2
    S_CO2_ave = 0.2 + Ks*(Solubility-0.015) + Kr*(1/C_Land-1)
    
    K_LCC = 1
    
    ##pre adjust 0.6
    pressureFront_vh_ratio = kvkh_ratio*(K_LCC*(LCC/0.7)**2.0) #####Verify
    
    ##pre adjust 9.0
    plumeVolume_to_pressureFrontVolume_ratio = 9.0 #####Verify
    
    
    #simulation step
    
    
    Pe0 = Datum*Pressure_gradient ## 150 bar in Pa for datum -1500m
#    print("Initial pressure in Pa, Pe0 = " + str(Pe0) )
    
    
    # J = 1.8E-3*kh ## in Reservoir Barrels / day /psi ; kh in mD-ft
    # J = 1.8E-3*119.25*kh ## in Reservoir kg / day /psi ; kh in mD-ft
    # J = 1.8E-3*119.25/(3600*24)*kh ## in Reservoir kg / s /psi ; kh in mD-ft
    # J = 1.8E-3*119.25/(3600*24*6894.76)*kh ## in Reservoir kg / s /Pa ; kh in mD-ft
    # J = 1.8E-3*119.25/(3600*24*6894.76)*k/md_to_m2*h*3.28084 ## in Reservoir kg / s /Pa ; kh in m2-m
    
    Jm_singleWell = 1.8E-3*119.25/(3600*24*6894.76)*k/mD_to_m2*(injection_interval*NTG)*3.28084 ## in Reservoir kg / s /Pa ; kh in m2-m
    
    Pwf_orig = Pwf
    
    #Jm = well_count*1/(75*1e5)*1e9/year_in_seconds ##12 wells, of 1 miton/year each for a deltaP of 75 bar, converted to kg/sec  
    Jm = well_count*Jm_singleWell
    
    if(Jm*(Pwf-Pe0)>maximumRate):
        Pwf = Pe0+maximumRate/Jm
    
    geometryStrike = squaredArea**0.5 
    geometryDip = squaredArea**0.5 
    geometryThickness = 1.5e3 #1.0 km at the core
    geometryAqThickness = 2*geometryThickness #2 km at the whole hydraulic porous space
    
 #   print("General injectivity in (kg/s)/Pa, Jm = " + str(Jm) )
    viscw = 0.6e-3
    Jo = (2*geometryStrike+2*geometryDip)*geometryAqThickness*k/viscw*NTG*phi/((0.5*geometryStrike+0.5*geometryDip)/2) ## (10+10+5+5)*3.0*1e6 perimeter area in m2 ; k perm in m2
 #   print("General aquifer dispersion flux in (m3/s)/Pa, Jo = " + str(Jo) )
    
    Va = geometryStrike*geometryDip*geometryThickness*phi*NTG #20x50x1.5km, 0.2 porosity, 0.5 NTG, in m
 #   print("Injection area Aquifer Volume in m3, Va = " + str(Va) )
    
    VA = geometryStrike*geometryDip*geometryAqThickness*phi*NTG #20x20x2km, 0.2 porosity, 0.5 NTG, in m
 #   print("Outer Aquifer Volume in m3, VA = " + str(VA) )
    
    rhoCO2 = 750 ##kg/m3
 #   print("CO2 average density in kg/m3, rhoCO2 = " + str(rhoCO2) )
    
    
    
  ##  scaling_model = 0 ## verify or reformulate Pe for not constant Pwf (more generally discrete computation)
  ##  s = 1e-5 # scaling/damage in Pa per kg injected
        
    Pe = np.zeros(len(t))
    PA = np.zeros(len(t))
    dotm_i = np.zeros(len(t))
    
    
##    cf = 1.78e-5/(phi**0.4358)*(1/6.89476e-6)*(1/1e9)

    cf = (-5490.6*phi**3+ 4725.5*phi**2 - 1392.1*phi +153.3)*1e-6/6894.76
    cw = 4.4e-10
    Sw = 1.0
    ct = cf+Sw*cw
    
#     a11 = Jo/(Va*ct)*(-(1+Jm/(Jo*rhoCO2)))
#     a12 = Jo/(Va*ct)
#     a21 = Jo/(Va*ct)*Va/VA
#     a22 = Jo/(Va*ct)*-Va/VA

# ##    a11 = -Pe0/Va*(Jm/rhoCO2+Jo)
# ##    a12 = Pe0*Jo/Va
# ##    a21 = Pe0*Jo/VA
# ##    a22 = -Pe0*Jo/VA
    
#     A = np.array([
#         [a11,a12],
#         [a21,a22]
#         ])
    
#     eigenvalues, eigenvectors = np.linalg.eig(A)
    
#     lambda1 = eigenvalues[0]
#     lambda2 = eigenvalues[1]
    
#     v11 = eigenvectors[0,0]
#     v12 = eigenvectors[1,0]
#     v21 = eigenvectors[0,1]
#     v22 = eigenvectors[1,1]
    
#     v = np.array([
#         [v11,v21],
#         [v12,v22]
#         ])
    
#     v_inv = np.linalg.inv(v)
    
#     C1 = v_inv[0,0]*(Pe0-Pwf)+v_inv[0,1]*(Pe0-Pwf)
#     C2 = v_inv[1,0]*(Pe0-Pwf)+v_inv[1,1]*(Pe0-Pwf)
    

#    Pe = Pwf+C1*np.exp(lambda1*t)*v11+C2*np.exp(lambda2*t)*v21
#    PA = Pwf+C1*np.exp(lambda1*t)*v12+C2*np.exp(lambda2*t)*v22


    PA[0]=Pe0
    Pe[0]=Pe0
#    if scaling_model == 0:
    dotm_i[0] = Jm*(Pwf-Pe[0])
    # else:
    #     S=0
    #     dt = (t[-1]-t[0])/len(t)
    #     dotm_i[0] = Jm*(Pwf-S-Pe[0])
    #     S = S + s*dt*dotm_i[0]

    i = 1
    while(i<len(t)):
        # if scaling_model == 0:

        Pwf = Pwf_orig
        if(Jm*(Pwf-Pe[i-1])>maximumRate):
            Pwf = Pe[i-1]+maximumRate/Jm

            
        dotm_i[i] = Jm*(Pwf-Pe[i-1])
        # else:
        #     dt = (t[-1]-t[0])/len(t)
        #     dotm_i[i] = Jm*(Pwf-S-Pe[i-1])
        #     S = S + s*dt*dotm_i[i]
    
        PA[i]=PA[i-1]+1/(VA*ct)*Jo*(Pe[i-1]-PA[i-1])*(t[i]-t[i-1])
        Pe[i]=Pe[i-1]+1/(Va*ct)*((dotm_i[i-1]/rhoCO2)*(t[i]-t[i-1])-Jo*(Pe[i-1]-PA[i-1])*(t[i]-t[i-1]))
        if not np.isfinite(PA[i]):
            print(f"Overflow at iteration {i}: VA={VA}, ct={ct}, Jo={Jo}, Pe={Pe[i-1]}, PA={PA[i-1]}")
            break

        i=i+1
    
    
    m_i = cumulative_trapezoid(dotm_i, t, initial=0)
    
    
    
    CO2_plume_bulkrockvolume = (m_i/rhoCO2)/(phi*NTG*S_CO2_ave)
    CO2_pressureFront_bulkrockvolume = plumeVolume_to_pressureFrontVolume_ratio*CO2_plume_bulkrockvolume
    
    
    ## ah = v
    ## a=pi r^2 ; r/h = ratio
    ## h = r/ratio = (a/pi)^0.5/ratio
    ## a(a/pi)^0.5 = v*ratio
    ## a^1.5 = v*ratio*pi^0.5

 #   CO2_plumeArea = (np.clip(CO2_plume_bulkrockvolume,0,100e9)*(np.pi)**0.5*plume_vh_ratio)**(2/3)
    CO2_pressureFrontArea = (np.clip(CO2_pressureFront_bulkrockvolume,0,1000e9)*(np.pi)**0.5*pressureFront_vh_ratio)**(2/3) ## correlation between pressure Area and LCC 
    
    CO2_plumeHeight=np.zeros(len(t))
    CO2_plumeHeight[0]=0.0
 #   CO2_plumeHeight[1:]= CO2_plume_bulkrockvolume[1:] / CO2_plumeArea[1:]

    CO2_pressureFrontHeight=np.zeros(len(t))
    CO2_pressureFrontHeight[0]=0.0
    CO2_pressureFrontHeight[1:]= CO2_pressureFront_bulkrockvolume[1:] / CO2_pressureFrontArea[1:]
    
 #   print(f"RESULT: cumulative volume injected in Mi tons = {m_i[-1]/1e9:.2f}")
 #   print(f"RESULT: average mass injected during first 5 years, in Mi tons / year = {dotm_i_ave5y/1e9*year_in_seconds:.2f}")
 #   print(f"RESULT: Plume area in km^2= {CO2_plumeArea[-1]/1e6:.2f}")
 #   print(f"RESULT: Plume height in m= {CO2_plumeHeight[-1]:.2f}")
 #   print(f"RESULT: Shallowest CO2 in m deep = {shallowest_CO2[-1]:.2f}")
    
    
 #   print(f"RESULT: Plume area in km^2 after 200 miton injected= {CO2_plumeArea[index]/1e6:.2f}")
 #   print(f"RESULT: Plume height in m after 200 miton injected= {CO2_plumeHeight[index]:.2f}")
    
 #   print(f"Final pressure P [bar]= {Pe[-1]/1e5:.2f}")
    
#    plt.scatter((t/year_in_seconds),(Pe/1e5))
    #plt.scatter((t/year_in_seconds),(PA/1e5))
#    plt.scatter((t/year_in_seconds),(dotm_i/1e9*year_in_seconds))
    #plt.scatter((t/year_in_seconds),(dotm_i/1e9))
    #print(dotm_i_ave5y/1e9*year_in_seconds)
    #print(m_i[-1]/1e9)
    
 #   plt.show()
    
    ########Compute the localized overpressure
    
    localized_H_ratio = 2
    subVa = Va/localized_H_ratio**2
    VA = VA+(Va-subVa)
    Va = subVa
    Jo = Jo/localized_H_ratio**2
    
    
    cf = (-5490.6*phi**3+ 4725.5*phi**2 - 1392.1*phi +153.3)*1e-6/6894.76
    cw = 4.4e-10
    Sw = 1.0-S_CO2_ave
    cCO2 = 3.65e-8
    SCO2 = S_CO2_ave
    ct = cf+Sw*cw+SCO2*cCO2

    
    a11 = Jo/(Va*ct)*(-(1+Jm/(Jo*rhoCO2)))
    a12 = Jo/(Va*ct)
    a21 = Jo/(Va*ct)*Va/VA
    a22 = Jo/(Va*ct)*-Va/VA

    
    A = np.array([
        [a11,a12],
        [a21,a22]
        ])
    
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    lambda1 = eigenvalues[0]
    lambda2 = eigenvalues[1]
    
    v11 = eigenvectors[0,0]
    v12 = eigenvectors[1,0]
    v21 = eigenvectors[0,1]
    v22 = eigenvectors[1,1]
    
    v = np.array([
        [v11,v21],
        [v12,v22]
        ])
    
    v_inv = np.linalg.inv(v)
    
    C1 = v_inv[0,0]*(Pe0-Pwf)+v_inv[0,1]*(Pe0-Pwf)
    C2 = v_inv[1,0]*(Pe0-Pwf)+v_inv[1,1]*(Pe0-Pwf)
    
    Pe = Pwf+C1*np.exp(lambda1*t)*v11+C2*np.exp(lambda2*t)*v21
    
    
    ########Compute the pressure front
    
    V_maximum_spread_pressure = 2*(injection_base - injection_interval/2 - SeaWaterLevel) ## pressure spreads to the top and to the bottom
    H_maximum_spread_pressure = V_maximum_spread_pressure*pressureFront_vh_ratio
    
    H_expected_spread_pressure = (CO2_pressureFrontArea**0.5/np.pi**0.5) ###Verify
    
    H_spread_pressure = np.minimum(H_expected_spread_pressure,H_maximum_spread_pressure)
    V_spread_pressure = H_spread_pressure/pressureFront_vh_ratio
    
    pressure_front_topmost_limit = injection_base-injection_interval/2-V_spread_pressure/2
    
    #exponential decay with 5 bar limiting the pressure front
    
    pressure_on_bottom_last_formation_barrier = np.zeros(len(t))
    pressure_on_bottom_last_formation_barrier[0] = 0
    
    transitionZonePressure = 100
    
    i = 1 
    while (i<len(t)):
        if(Pe[i]-Pe0>5e5): ## avoid expurious log extrapolation
            x1, y1 = pressure_front_topmost_limit[i]+transitionZonePressure, Pe[i]-Pe0
            x2, y2 = pressure_front_topmost_limit[i]-transitionZonePressure, 5e5 ##5 bar limit for pressure front
            x_target = bottom_last_formation_barrier
            
            # Solve for B
            B = np.log(y1 / y2) / (x1 - x2)
            # Solve for A
            A = y1 / np.exp(B * x1)
            # Compute interpolated value
            y_target = A * np.exp(B * x_target)
    
            pressure_on_bottom_last_formation_barrier[i] = y_target
    ##        print(f"CALC: x1={x1}, x2={x2}, y1={y1}, y2={y2}, x_target={x_target}, y_target={y_target}")
            if not np.isfinite(A):
                print(f"Overflow: x1={x1}, x2={x2}, y1={y1}, y2={y2}")
                break
        else:
            pressure_on_bottom_last_formation_barrier[i] = 0
        i=i+1
    
    overpressure_coreArea_percent = (Pe-Pe0)/(injection_base*Pressure_gradient)
    pressure_on_bottom_last_formation_barrier_percent = pressure_on_bottom_last_formation_barrier/(bottom_last_formation_barrier*Pressure_gradient)
    
#    print(f"RESULT: Pressure front area, in km2 = {H_spread_pressure[-1]**2/1e6*np.pi:.2f}")
#    print(f"RESULT: Pressure front height, in m  = {V_spread_pressure[-1]:.2f}")
#    print(f"RESULT: CO2 plume height, in m = {CO2_plumeHeight[-1]:.2f}")
    
#    print(f"RESULT: Overpressure in the core area, in bar = {overpressure_coreArea[-1]/1e5:.2f}")
#    print(f"RESULT: Overpressure in the bottom of last formation (limit horizont), in bar = {pressure_on_bottom_last_formation_barrier[-1]/1e5:.2f}")
    
#    print(f"RESULT: Overpressure in the core area, in % of original pressure = {overpressure_coreArea_percent[-1]*100:.2f}%")
#    print(f"RESULT: Overpressure in the bottom of last formation (limit horizont), in % of original pressure = {pressure_on_bottom_last_formation_barrier_percent[-1]*100:.2f}%")
    

    
    ## results restricted to geomec 
    
    maxPercentage = (geomecGradient_deeper*percentage_geomecLimits_deeper - Pressure_gradient)/Pressure_gradient
    effectiveSafetyMargin_coreArea = (maxPercentage - overpressure_coreArea_percent)/maxPercentage
    
    if(effectiveSafetyMargin_coreArea[-1]>0):
        index_1 = len(m_i)
        capacity_restricted_geomec_coreAreaPressure = m_i[-1]
    else:
        index_1 = np.abs(effectiveSafetyMargin_coreArea).argmin()
        capacity_restricted_geomec_coreAreaPressure = m_i[index_1]
    
#    print(f"RESULT: Capacity restricted to geomecanical limits on central area in Miton = {capacity_restricted_geomec_coreAreaPressure/1e9:.2f}")
    
    presSeal = pressure_on_bottom_last_formation_barrier + bottom_last_formation_barrier * Pressure_gradient
    presLimit =  (geomecGradient_shallow * percentage_geomecLimits_shallow) * bottom_last_formation_barrier

    effectiveSafetyMargin_lastFormation = presLimit-presSeal
    
    if(effectiveSafetyMargin_lastFormation[-1]>0):
        index_2 = len(m_i)
        capacity_restricted_geomec_lastFormationBarrierPressure  = m_i[-1]
    else:
        index_2 = np.abs(effectiveSafetyMargin_lastFormation).argmin()
        capacity_restricted_geomec_lastFormationBarrierPressure  = m_i[index_2]
    
    
 #   print(f"RESULT: Capacity restricted to geomecanical limits on the bottom of last barrier, in Miton = {capacity_restricted_geomec_lastFormationBarrierPressure/1e9:.2f}")
    
    
    capacity_restrictions = np.min([capacity_restricted_geomec_coreAreaPressure,capacity_restricted_geomec_lastFormationBarrierPressure])
  
    print(capacity_restricted_geomec_coreAreaPressure)
    print(capacity_restricted_geomec_lastFormationBarrierPressure)
  
 #   print(f"RESULT: Capacity considering geomec restrictions, in Miton = {capacity_restrictions/1e9:.2f} (vs. {m_i[-1]/1e9:.2f} unrestricted)")

    capacity_restrictions = capacity_restrictions+1####adding something to avoid log problems

    #### plume geometry code
    plume_vh_ratio = pressureFront_vh_ratio
    CO2_plumeArea = (np.clip(CO2_plume_bulkrockvolume,0,1000e9)*(np.pi)**0.5*plume_vh_ratio)**(2/3)
    CO2_equivalentRadius = (CO2_plumeArea/np.pi)**0.5
    
    CO2_plumeHeight=np.zeros(len(t))
    CO2_plumeHeight[0]=0.0
    CO2_plumeHeight[1:]= CO2_plume_bulkrockvolume[1:] / np.maximum(CO2_plumeArea,squaredArea)[1:]
    CO2_plumeHeight[1:]=np.maximum(CO2_plumeHeight[1:],injection_interval)


    
    return CO2_equivalentRadius,CO2_plumeHeight,H_spread_pressure,V_spread_pressure,pressure_on_bottom_last_formation_barrier,geomecGradient_shallow,percentage_geomecLimits_shallow,overpressure_coreArea_percent,percentage_geomecLimits_deeper,t,dotm_i,m_i,(capacity_restrictions)#capacity_restrictions

plt.close('all') 



############ Constants
Datum = 1500 # m
bottom_last_formation_barrier = 1200 ##m
#print("Bottom of topmost formation (last barrier), in m = " + str(bottom_last_formation_barrier) )
SeaWaterLevel = 400 ##m
#print("Mean Sea Water Level, in m = " + str(SeaWaterLevel) )

############ Natural inputs with uncertainty

phi = 0.16 
k = 27*mD_to_m2  # keffective (krel*kabs) log-normal
kvkh_ratio = 10
LCC = 0.7
C_Land = 1 #log-normal
Solubility = 0.015 ##average Rs of CO2 in brine
Pressure_gradient = 0.102e5 ## Pa/m
geomecGradient_shallow = 0.154e5
geomecGradient_deeper = 0.139e5

############ Designed (controllable) inputs

Pwf = 210*1e5 ## 210 bar in Pa at datum -1500m
#print("Injection pressure in Pa, Pwf = " + str(Pwf) )
well_count = 12
injection_base = 1900 
injection_interval = 200
maximumRate = 30*1e9/year_in_seconds # maximumRate of 30 miton/year, effective as a restriction for too big injectivities

#design constants
maximumAllowedPlumeReach = 10000 #plume allowed to migrate within 10km mean radius
minimumAllowedLevel = bottom_last_formation_barrier + 200 #safety margin of 200m from bottom of last formation barrier

safetyPercentage_geomecLimits_shallow = 0.8
safetyPercentage_geomecLimits_deeper = 0.9

squaredArea = 12e3*9e3 #12kmx9km injection acreage

designTarget = 200e9
designTargetReached = 0


## model run
#defining the input
mu_x_in = [phi,np.log(k),kvkh_ratio,LCC,np.log(C_Land),Solubility,Pressure_gradient,geomecGradient_shallow,geomecGradient_deeper]

#def proxy_model_CCS(Datum,bottom_last_formation_barrier,SeaWaterLevel,phi,k,kvkh_ratio,LCC,C_Land,Solubility,Pressure_gradient,geomecGradient_shallow,geomecGradient_deeper,Pwf,well_count,injection_base,injection_interval,maximumRate,maximumAllowedPlumeReach,minimumAllowedLevel,percentage_geomecLimits_shallow,percentage_geomecLimits_deeper,squaredArea):

# Streamlit
st.title("CO2 plume and pressure front")
# Initialize flags
if "run_simulation" not in st.session_state:
    st.session_state["run_simulation"] = False
if "prev_inputs" not in st.session_state:
    st.session_state["prev_inputs"] = {}
if "running" not in st.session_state:
    st.session_state["running"] = False


# Sliders with keys
st.slider("Uncertainty parameter: Select a value for LCC", 0.2, 1.0, 0.7, key="valueLCC_input")
st.slider("Uncertainty parameter: Select a value for Permeability (mD)", 1, 200, 27, key="valuek_input")
st.slider("Design parameter: Select topside injection pressure (bar)", 10, 200, 75, key="valuePwf_input")
st.slider("Design parameter: Select Maximum rate on cluster (Million t/year)", 3, 50, 20, key="valueMaxRate_input")

# Button to trigger simulation
clicked = st.button("▶ Run Simulation", disabled=st.session_state["running"])
if clicked:
    st.session_state["run_simulation"] = True
    st.session_state["running"] = True


# Detect if any slider was changed -> cancel run_simulation
current_inputs = {
    "LCC": st.session_state["valueLCC_input"],
    "k": st.session_state["valuek_input"],
    "Pwf": st.session_state["valuePwf_input"],
    "MaxRate": st.session_state["valueMaxRate_input"]
}

if current_inputs != st.session_state["prev_inputs"]:
    st.session_state["run_simulation"] = False
    st.session_state["prev_inputs"] = current_inputs

# Run simulation only when triggered
if st.session_state["run_simulation"]:
    time.sleep(.2)

    with st.spinner("Generating animation, please wait..."):    
        CO2_equivalentRadius, CO2_plumeHeight, H_spread_pressure, V_spread_pressure, \
        pressure_on_bottom_last_formation_barrier, geomecGradient_shallow, \
        percentage_geomecLimits_shallow, overpressure_coreArea_percent, \
        percentage_geomecLimits_deeper, t, dotm_i, m_i, BaseCase = proxy_model_CCS(
            Datum=Datum,
            bottom_last_formation_barrier=bottom_last_formation_barrier,
            SeaWaterLevel=SeaWaterLevel,
            phi=mu_x_in[0],
            k=st.session_state["valuek_input"] * mD_to_m2,
            kvkh_ratio=mu_x_in[2],
            LCC=st.session_state["valueLCC_input"],
            C_Land=np.exp(mu_x_in[4]),
            Solubility=mu_x_in[5],
            Pressure_gradient=mu_x_in[6],
            geomecGradient_shallow=mu_x_in[7],
            geomecGradient_deeper=mu_x_in[8],
            Pwf=(150 + st.session_state["valuePwf_input"]) * 1e5,
            well_count=well_count,
            injection_base=injection_base,
            injection_interval=injection_interval,
            maximumRate=st.session_state["valueMaxRate_input"] * 1e9 / year_in_seconds,
            maximumAllowedPlumeReach=maximumAllowedPlumeReach,
            minimumAllowedLevel=minimumAllowedLevel,
            percentage_geomecLimits_shallow=safetyPercentage_geomecLimits_shallow,
            percentage_geomecLimits_deeper=safetyPercentage_geomecLimits_deeper,
            squaredArea=squaredArea,
        )
    
        if (BaseCase < m_i[-1]):
            st.write(f"Capacity: {BaseCase/1e9:.2f} Million tons (constrained by geomechanics)")
        else: 
            st.write(f"Capacity under restrictions: {BaseCase/1e9:.2f} Million tons (no limit reached)")
        video_file = generate_simulation_animation(
            CO2_equivalentRadius, CO2_plumeHeight, H_spread_pressure, V_spread_pressure,
            pressure_on_bottom_last_formation_barrier, geomecGradient_shallow,
            percentage_geomecLimits_shallow, overpressure_coreArea_percent,
            percentage_geomecLimits_deeper, t, dotm_i, m_i,BaseCase
        )
    
    ################ainda com um problema no limite geomecanico do intervalo raso. na figura ele pinta de vermelho corretamente, mas no calculo do volume maximo esta com problema.....
    
        st.video(f"{video_file}")
     
        st.session_state["running"] = False
