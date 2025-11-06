import SimpleITK as sitk

def dice_coeff(pred, target, e = 1e-6):
    return

def show_slices(slices, titles=None):
    return

'''
Registers the moving_img to fixed_img. Same code from Lab B Pt 4, but sampling percentage dropped to 0.01 for speed
Input: 2x SimpleITK image
Output: 1x SimpleITK image of the transformed moving_img
'''
def registration (fixed_img, moving_img):
    min_value = float(sitk.GetArrayViewFromImage(moving_img).min())
    init_tx = sitk.CenteredTransformInitializer(
        fixed_img, moving_img,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.01)
    R.SetInterpolator(sitk.sitkLinear)

    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-4, numberOfIterations=500, relaxationFactor=0.5
    )

    R.SetOptimizerScalesFromPhysicalShift()

    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    R.SetInitialTransform(init_tx, inPlace=False)

    final_tx = R.Execute(fixed_img, moving_img)

    transformed_img = sitk.Resample(
        moving_img, 
        fixed_img, 
        final_tx,
        sitk.sitkLinear, min_value,
        moving_img.GetPixelIDValue()
    )

    return transformed_img

