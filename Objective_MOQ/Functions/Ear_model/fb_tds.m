function [ Ref, Test ] = fb_tds( E0_ref, E0_test, General, match_method )
%[ E1 ] = fb_tds_backwards( E0 )
%   Time Domain Smearing (1) Backward masking Equation 35
%   Adding of Internal Noise Equation 36,37
%   Time Domain Smearing (2) Forward masking Equation 38-40
global debug_var

if debug_var
    fprintf('  Backwards Time Domain Smearing & Internal Noise\n')
end

[ E2_ref, E_ref ] = fb_tds_calc( E0_ref, General );
[ E2_test, E_test ] = fb_tds_calc( E0_test, General );

%Time-Align the signals
%For now, only align to the length of the test signal

switch match_method
    case 'Framing_Test'
        E_ref_interp = zeros(size(E_test));
        for k = 1:size(E_test,1)
            E_ref_interp(k,:) = interp1(linspace(0,1,size(E_ref,2)),E_ref(k,:),linspace(0,1,size(E_ref_interp,2)));
        end

        E2_ref_interp = zeros(size(E2_test));
        for k = 1:size(E2_test,1)
            E2_ref_interp(k,:) = interp1(linspace(0,1,size(E2_ref,2)),E2_ref(k,:),linspace(0,1,size(E2_ref_interp,2)));
        end
        Ref.E = E_ref_interp.';
        Ref.E2 = E2_ref_interp.';
        Test.E = E_test.';
        Test.E2 = E2_test.';
    case 'Framing_Ref'
        E_test_interp = zeros(size(E_ref));
        for k = 1:size(E_ref,1)
            E_test_interp(k,:) = interp1(linspace(0,1,size(E_test,2)),E_test(k,:),linspace(0,1,size(E_test_interp,2)));
        end
        E2_test_interp = zeros(size(E2_ref));
        for k = 1:size(E2_ref,1)
            E2_test_interp(k,:) = interp1(linspace(0,1,size(E2_test,2)),E2_test(k,:),linspace(0,1,size(E2_test_interp,2)));
        end
        Ref.E = E_ref_interp.';
        Ref.E2 = E2_ref_interp.';
        Test.E = E_test.';
        Test.E2 = E2_test.';
    case 'Interpolate_to_test'
        E_ref_interp = zeros(size(E_test));
        for k = 1:size(E_test,1)
            E_ref_interp(k,:) = interp1(linspace(0,1,size(E_ref,2)),E_ref(k,:),linspace(0,1,size(E_ref_interp,2)));
        end

        E2_ref_interp = zeros(size(E2_test));
        for k = 1:size(E2_test,1)
            E2_ref_interp(k,:) = interp1(linspace(0,1,size(E2_ref,2)),E2_ref(k,:),linspace(0,1,size(E2_ref_interp,2)));
        end
        Ref.E = E_ref_interp.';
        Ref.E2 = E2_ref_interp.';
        Test.E = E_test.';
        Test.E2 = E2_test.';
    case 'Interpolate_to_ref'
        E_test_interp = zeros(size(E_ref));
        for k = 1:size(E_ref,1)
            E_test_interp(k,:) = interp1(linspace(0,1,size(E_test,2)),E_test(k,:),linspace(0,1,size(E_test_interp,2)));
        end
        E2_test_interp = zeros(size(E2_ref));
        for k = 1:size(E2_ref,1)
            E2_test_interp(k,:) = interp1(linspace(0,1,size(E2_test,2)),E2_test(k,:),linspace(0,1,size(E2_test_interp,2)));
        end
        Ref.E = E_ref.';
        Ref.E2 = E2_ref.';
        Test.E = E_test_interp.';
        Test.E2 = E2_test_interp.';
      case 'Interpolate_fd_down'
          % Interpolating to test
          E_ref_interp = zeros(size(E_test));
          for k = 1:size(E_test,1)
              E_ref_interp(k,:) = interp1(linspace(0,1,size(E_ref,2)),E_ref(k,:),linspace(0,1,size(E_ref_interp,2)));
          end

          E2_ref_interp = zeros(size(E2_test));
          for k = 1:size(E2_test,1)
              E2_ref_interp(k,:) = interp1(linspace(0,1,size(E2_ref,2)),E2_ref(k,:),linspace(0,1,size(E2_ref_interp,2)));
          end
          Ref.E = E_ref_interp.';
          Ref.E2 = E2_ref_interp.';
          Test.E = E_test.';
          Test.E2 = E2_test.';
      case 'Interpolate_fd_up'
      % Interpolating to test
          E_ref_interp = zeros(size(E_test));
          for k = 1:size(E_test,1)
              E_ref_interp(k,:) = interp1(linspace(0,1,size(E_ref,2)),E_ref(k,:),linspace(0,1,size(E_ref_interp,2)));
          end

          E2_ref_interp = zeros(size(E2_test));
          for k = 1:size(E2_test,1)
              E2_ref_interp(k,:) = interp1(linspace(0,1,size(E2_ref,2)),E2_ref(k,:),linspace(0,1,size(E2_ref_interp,2)));
          end
          Ref.E = E_ref_interp.';
          Ref.E2 = E2_ref_interp.';
          Test.E = E_test.';
          Test.E2 = E2_test.';
    otherwise
        disp('Unknown Match method.  Interpolating to test.')
        E_ref_interp = zeros(size(E_test));
        for k = 1:size(E_test,1)
            E_ref_interp(k,:) = interp1(linspace(0,1,size(E_ref,2)),E_ref(k,:),linspace(0,1,size(E_ref_interp,2)));
        end

        E2_ref_interp = zeros(size(E2_test));
        for k = 1:size(E2_test,1)
            E2_ref_interp(k,:) = interp1(linspace(0,1,size(E2_ref,2)),E2_ref(k,:),linspace(0,1,size(E2_ref_interp,2)));
        end
        Ref.E = E_ref_interp.';
        Ref.E2 = E2_ref_interp.';
        Test.E = E_test.';
        Test.E2 = E2_test.';
end




end
