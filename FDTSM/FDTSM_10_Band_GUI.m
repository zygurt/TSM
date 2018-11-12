function varargout = FDTSM_10_Band_GUI(varargin)
% FDTSM_10_BAND_GUI MATLAB code for FDTSM_10_Band_GUI.fig
%      FDTSM_10_BAND_GUI, by itself, creates a new FDTSM_10_BAND_GUI or raises the existing
%      singleton*.
%
%      H = FDTSM_10_BAND_GUI returns the handle to a new FDTSM_10_BAND_GUI or the handle to
%      the existing singleton*.
%
%      FDTSM_10_BAND_GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in FDTSM_10_BAND_GUI.M with the given input arguments.
%
%      FDTSM_10_BAND_GUI('Property','Value',...) creates a new FDTSM_10_BAND_GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before FDTSM_10_Band_GUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to FDTSM_10_Band_GUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help FDTSM_10_Band_GUI

% Last Modified by GUIDE v2.5 26-Mar-2018 20:15:35

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @FDTSM_10_Band_GUI_OpeningFcn, ...
                   'gui_OutputFcn',  @FDTSM_10_Band_GUI_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before FDTSM_10_Band_GUI is made visible.
function FDTSM_10_Band_GUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to FDTSM_10_Band_GUI (see VARARGIN)

% Choose default command line output for FDTSM_10_Band_GUI
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes FDTSM_10_Band_GUI wait for user response (see UIRESUME)
 uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = FDTSM_10_Band_GUI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
output_arg.TSM_ratios = [get(handles.slider1,'Value'), ...
                         get(handles.slider2,'Value'), ...
                         get(handles.slider3,'Value'), ...
                         get(handles.slider4,'Value'), ...
                         get(handles.slider5,'Value'), ...
                         get(handles.slider6,'Value'), ...
                         get(handles.slider7,'Value'), ...
                         get(handles.slider8,'Value'), ...
                         get(handles.slider9,'Value'), ...
                         get(handles.slider10,'Value')];
output_arg.filename = get(handles.text_filename,'String');
output_arg.path = get(handles.text_path,'String');
%output_arg.output_name = get(handles.edit_text_output_name,'String');
varargout{1} = output_arg;
close(handles.figure1);


% --- Executes on slider movement.
function slider1_Callback(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
slider_Value = get(handles.slider1,'Value');
set(handles.text_tsm1,'String',sprintf('%.2f%%',slider_Value*100));

% --- Executes during object creation, after setting all properties.
function slider1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on slider movement.
function slider2_Callback(hObject, eventdata, handles)
% hObject    handle to slider2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
slider_Value = get(handles.slider2,'Value');
set(handles.text_tsm2,'String',sprintf('%.2f%%',slider_Value*100));

% --- Executes during object creation, after setting all properties.
function slider2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on slider movement.
function slider3_Callback(hObject, eventdata, handles)
% hObject    handle to slider3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
slider_Value = get(handles.slider3,'Value');
set(handles.text_tsm3,'String',sprintf('%.2f%%',slider_Value*100));


% --- Executes during object creation, after setting all properties.
function slider3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on slider movement.
function slider4_Callback(hObject, eventdata, handles)
% hObject    handle to slider4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
slider_Value = get(handles.slider4,'Value');
set(handles.text_tsm4,'String',sprintf('%.2f%%',slider_Value*100));


% --- Executes during object creation, after setting all properties.
function slider4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on slider movement.
function slider5_Callback(hObject, eventdata, handles)
% hObject    handle to slider5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
slider_Value = get(handles.slider5,'Value');
set(handles.text_tsm5,'String',sprintf('%.2f%%',slider_Value*100));


% --- Executes during object creation, after setting all properties.
function slider5_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

% --- Executes on slider movement.
function slider6_Callback(hObject, eventdata, handles)
% hObject    handle to slider6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
slider_Value = get(handles.slider6,'Value');
set(handles.text_tsm6,'String',sprintf('%.2f%%',slider_Value*100));


% --- Executes during object creation, after setting all properties.
function slider6_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end




% --- Executes on slider movement.
function slider7_Callback(hObject, eventdata, handles)
% hObject    handle to slider7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
slider_Value = get(handles.slider7,'Value');
set(handles.text_tsm7,'String',sprintf('%.2f%%',slider_Value*100));


% --- Executes during object creation, after setting all properties.
function slider7_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

% --- Executes on slider movement.
function slider8_Callback(hObject, eventdata, handles)
% hObject    handle to slider8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
slider_Value = get(handles.slider8,'Value');
set(handles.text_tsm8,'String',sprintf('%.2f%%',slider_Value*100));


% --- Executes during object creation, after setting all properties.
function slider8_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on slider movement.
function slider9_Callback(hObject, eventdata, handles)
% hObject    handle to slider9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
slider_Value = get(handles.slider9,'Value');
set(handles.text_tsm9,'String',sprintf('%.2f%%',slider_Value*100));

% --- Executes during object creation, after setting all properties.
function slider9_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on slider movement.
function slider10_Callback(hObject, eventdata, handles)
% hObject    handle to slider10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
slider_Value = get(handles.slider10,'Value');
set(handles.text_tsm10,'String',sprintf('%.2f%%',slider_Value*100));

% --- Executes during object creation, after setting all properties.
function slider10_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

% --- Executes on button press in pushbutton_reset.
function pushbutton_reset_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_reset (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.slider1,'Value',1);
set(handles.text_tsm1,'String',sprintf('%.2f%%',100));

set(handles.slider2,'Value',1);
set(handles.text_tsm2,'String',sprintf('%.2f%%',100));

set(handles.slider3,'Value',1);
set(handles.text_tsm3,'String',sprintf('%.2f%%',100));

set(handles.slider4,'Value',1);
set(handles.text_tsm4,'String',sprintf('%.2f%%',100));

set(handles.slider5,'Value',1);
set(handles.text_tsm5,'String',sprintf('%.2f%%',100));

set(handles.slider6,'Value',1);
set(handles.text_tsm6,'String',sprintf('%.2f%%',100));

set(handles.slider7,'Value',1);
set(handles.text_tsm7,'String',sprintf('%.2f%%',100));

set(handles.slider8,'Value',1);
set(handles.text_tsm8,'String',sprintf('%.2f%%',100));

set(handles.slider9,'Value',1);
set(handles.text_tsm9,'String',sprintf('%.2f%%',100));

set(handles.slider10,'Value',1);
set(handles.text_tsm10,'String',sprintf('%.2f%%',100));


% --- Executes on button press in pushbutton_continue.
function pushbutton_continue_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_continue (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if(strcmp('File: Not Specified',get(handles.text_filename,'String')) == 0 )
    uiresume(handles.figure1);
end


% --- Executes on button press in pushbutton_open.
function pushbutton_open_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_open (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

[file,path] = uigetfile('*.wav');
set(handles.text_filename,'String',file);
set(handles.text_path,'String',path);
[~, FS] = audioread([path file]);

freq_middle_cell = {};
N = 2048;
bin_freq_width = FS/N;

Upper = [2.^(1:9) N/2+1];
Lower = [1 Upper(1:end-1)+1];

freq_lower = (Lower-1)*bin_freq_width;
freq_upper = Upper*bin_freq_width;
freq_middle = [bin_freq_width/2 ((freq_lower(2:end)-1)+(freq_upper(2:end)-1))/2];

%Display centre frequency of the region
% for n = 1:length(freq_middle)
%     if freq_middle(n)<1000
%         freq_middle_cell{n} = sprintf("%.0f Hz",freq_middle(n));
%     else
%         freq_middle_cell{n} = sprintf("%.1f kHz",freq_middle(n)/1000);
%     end
% end

%Display Upper and Lower frequency bounds of the region
for n = 1:length(freq_middle)
    if freq_middle(n)<1000
        freq_middle_cell{n} = sprintf('%.0f -\n%.0f Hz',freq_lower(n),freq_upper(n));
    else
        freq_middle_cell{n} = sprintf('%.1f -\n%.1f kHz',freq_lower(n)/1000,freq_upper(n)/1000);
    end
end

set(handles.text_freq1,'String',freq_middle_cell{1});
set(handles.text_freq2,'String',freq_middle_cell{2});
set(handles.text_freq3,'String',freq_middle_cell{3});
set(handles.text_freq4,'String',freq_middle_cell{4});
set(handles.text_freq5,'String',freq_middle_cell{5});
set(handles.text_freq6,'String',freq_middle_cell{6});
set(handles.text_freq7,'String',freq_middle_cell{7});
set(handles.text_freq8,'String',freq_middle_cell{8});
set(handles.text_freq9,'String',freq_middle_cell{9});
set(handles.text_freq10,'String',freq_middle_cell{10});
