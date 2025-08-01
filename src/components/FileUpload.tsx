import React, { useCallback, useState, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, File, Video, Music, Image, FileText, X, CheckCircle, AlertCircle } from 'lucide-react';
import { supabase } from '../lib/supabase';
import toast from 'react-hot-toast';

interface FileUploadProps {
  personaId: string;
  onUploadComplete?: (files: UploadedFile[]) => void;
}

interface UploadedFile {
  id: string;
  name: string;
  size: number;
  type: string;
  url: string;
  status: 'uploading' | 'processing' | 'completed' | 'error';
  progress: number;
  error?: string;
}

export function FileUpload({ personaId, onUploadComplete }: FileUploadProps) {
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [bucketError, setBucketError] = useState<string | null>(null);
  const [bucketChecked, setBucketChecked] = useState(false);

  useEffect(() => {
    checkAndCreateBucket();
  }, []);

  const isValidUUID = (id: string): boolean => {
    const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
    return uuidRegex.test(id);
  };

  const isPersonaIdValid = personaId && isValidUUID(personaId);

  const getFileIcon = (type: string) => {
    if (type.startsWith('video/')) return Video;
    if (type.startsWith('audio/')) return Music;
    if (type.startsWith('image/')) return Image;
    if (type.startsWith('text/') || type.includes('document')) return FileText;
    return File;
  };

  const getContentType = (mimeType: string): string => {
    if (mimeType.startsWith('video/')) return 'video';
    if (mimeType.startsWith('audio/')) return 'audio';
    if (mimeType.startsWith('image/')) return 'image';
    if (mimeType.startsWith('text/') || mimeType.includes('document')) return 'text';
    return 'text';
  };

  const checkAndCreateBucket = async (): Promise<boolean> => {
    try {
      setBucketChecked(false);
      
      // Instead of listing buckets, directly test bucket access with a small test operation
      try {
        const testPath = `test-${Date.now()}.txt`;
        const { error: uploadError } = await supabase.storage
          .from('persona-content')
          .upload(testPath, new Blob(['test']), {
            cacheControl: '3600',
            upsert: true
          });

        if (uploadError) {
          console.error('Test upload error:', uploadError);
          if (uploadError.message.includes('permission denied')) {
            setBucketError('Permission denied. Please check your storage bucket policies.');
          } else if (uploadError.message.includes('not found')) {
            setBucketError('Storage bucket "persona-content" not found. Please create it manually in your Supabase dashboard under Storage > New Bucket with a 50MB file size limit.');
          } else {
            setBucketError(`Storage access error: ${uploadError.message}`);
          }
          setBucketChecked(true);
          return false;
        }

        // Clean up test file
        await supabase.storage.from('persona-content').remove([testPath]);
      } catch (testError) {
        console.error('Bucket test error:', testError);
        setBucketError('Error testing bucket access. Please check your Supabase configuration.');
        setBucketChecked(true);
        return false;
      }

      setBucketError(null);
      setBucketChecked(true);
      return true;
    } catch (error) {
      console.error('Bucket check error:', error);
      setBucketError('Storage configuration error. Please check your Supabase setup.');
      setBucketChecked(true);
      return false;
    }
  };

  const uploadFile = async (file: File): Promise<UploadedFile> => {
    const fileId = `${Date.now()}-${file.name.replace(/[^a-zA-Z0-9.-]/g, '_')}`;
    const filePath = fileId;

    const uploadedFile: UploadedFile = {
      id: fileId,
      name: file.name,
      size: file.size,
      type: file.type,
      url: '',
      status: 'uploading',
      progress: 0
    };

    try {
      if (file.size > 50 * 1024 * 1024) {
        throw new Error('File size exceeds 50MB limit. Please choose a smaller file.');
      }

      if (!personaId || !isValidUUID(personaId)) {
        throw new Error('No valid persona selected. Please select a persona first.');
      }

      const bucketReady = await checkAndCreateBucket();
      if (!bucketReady) {
        throw new Error('Storage bucket not available');
      }

      console.log('Uploading file:', file.name, 'to path:', filePath);

      const { data: uploadData, error: uploadError } = await supabase.storage
        .from('persona-content')
        .upload(filePath, file, {
          cacheControl: '3600',
          upsert: true
        });

      if (uploadError) {
        console.error('Upload error details:', uploadError);
        throw new Error(`Upload failed: ${uploadError.message}`);
      }

      console.log('Upload successful:', uploadData);

      const { data: { publicUrl } } = supabase.storage
        .from('persona-content')
        .getPublicUrl(filePath);

      uploadedFile.url = publicUrl;
      uploadedFile.status = 'processing';
      uploadedFile.progress = 50;

      const dbPayload = {
        persona_id: personaId,
        content_type: getContentType(file.type),
        file_url: publicUrl,
        file_name: file.name,
        file_size: file.size,
        metadata: {
          original_name: file.name,
          mime_type: file.type,
          upload_date: new Date().toISOString()
        },
        processing_status: 'processing'
      };

      console.log('Saving to database with payload:', dbPayload);

      const { data: dbData, error: dbError } = await supabase
        .from('persona-content')
        .insert(dbPayload)
        .select();

      if (dbError) {
        console.error('Database error details:', {
          error: dbError,
          message: dbError.message,
          details: dbError.details,
          hint: dbError.hint,
          code: dbError.code,
          payload: dbPayload
        });
        
        let errorMessage = 'Database save failed';
        if (dbError.message) {
          errorMessage += `: ${dbError.message}`;
        } else if (dbError.details) {
          errorMessage += `: ${dbError.details}`;
        } else if (dbError.hint) {
          errorMessage += `: ${dbError.hint}`;
        } else {
          errorMessage += ': Unknown database error';
        }
        
        if (dbError.code === 'PGRST116') {
          errorMessage += ' (No rows returned - check RLS policies)';
        } else if (dbError.code === '23503') {
          errorMessage += ' (Foreign key violation - persona not found)';
        } else if (dbError.code === '42501') {
          errorMessage += ' (Permission denied - check RLS policies)';
        }
        
        throw new Error(errorMessage);
      }

      console.log('Database save successful:', dbData);

      setTimeout(() => {
        uploadedFile.status = 'completed';
        uploadedFile.progress = 100;
        setUploadedFiles(prev => 
          prev.map(f => f.id === fileId ? uploadedFile : f)
        );
        
        toast.success(`File "${file.name}" uploaded successfully!`);
      }, 2000);

      return uploadedFile;
    } catch (error) {
      console.error('Upload error:', error);
      uploadedFile.status = 'error';
      uploadedFile.progress = 0;
      uploadedFile.error = error instanceof Error ? error.message : 'Upload failed';
      
      toast.error(`Upload failed: ${uploadedFile.error}`);
      return uploadedFile;
    }
  };

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (!isPersonaIdValid) {
      toast.error('Please select a persona before uploading files');
      return;
    }

    if (!bucketChecked) {
      await checkAndCreateBucket();
    }
    
    setIsUploading(true);
    
    // Filter out files that are too large
    const validFiles = acceptedFiles.filter(file => {
      if (file.size > 50 * 1024 * 1024) {
        console.warn(`File ${file.name} is too large (${formatFileSize(file.size)}). Maximum size is 50MB.`);
        toast.error(`File ${file.name} is too large. Maximum size is 50MB.`);
        return false;
      }
      return true;
    });

    if (validFiles.length === 0) {
      setIsUploading(false);
      return;
    }
    
    const newFiles: UploadedFile[] = validFiles.map(file => ({
      id: `${Date.now()}-${file.name.replace(/[^a-zA-Z0-9.-]/g, '_')}`,
      name: file.name,
      size: file.size,
      type: file.type,
      url: '',
      status: 'uploading' as const,
      progress: 0
    }));

    setUploadedFiles(prev => [...prev, ...newFiles]);

    try {
      const uploadPromises = validFiles.map(uploadFile);
      const results = await Promise.all(uploadPromises);
      
      setUploadedFiles(prev => 
        prev.map(file => {
          const result = results.find(r => r.id === file.id);
          return result || file;
        })
      );

      onUploadComplete?.(results.filter(r => r.status === 'completed'));
    } catch (error) {
      console.error('Batch upload error:', error);
      toast.error('Some files failed to upload. Please try again.');
    } finally {
      setIsUploading(false);
    }
  }, [personaId, onUploadComplete, bucketChecked, isPersonaIdValid]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpg', '.jpeg', '.png', '.gif'],
      'video/*': ['.mp4', '.mov', '.avi', '.mkv'],
      'audio/*': ['.mp3', '.wav', '.m4a', '.aac'],
      'text/*': ['.txt', '.md'],
      'application/pdf': ['.pdf'],
      'application/msword': ['.doc'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx']
    },
    maxSize: 50 * 1024 * 1024, // 50MB
    multiple: true,
    disabled: !!bucketError || !isPersonaIdValid
  });

  const removeFile = (fileId: string) => {
    setUploadedFiles(prev => prev.filter(f => f.id !== fileId));
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="space-y-6">
      {/* Persona Selection Error Alert */}
      {!isPersonaIdValid && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <div className="flex items-start">
            <AlertCircle className="h-5 w-5 text-yellow-500 mt-0.5 mr-3 flex-shrink-0" />
            <div>
              <h4 className="text-yellow-800 font-medium">No Persona Selected</h4>
              <p className="text-yellow-700 text-sm mt-1">
                Please select a persona from your dashboard before uploading content. 
                Content must be associated with a specific persona for training.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Storage Error Alert */}
      {bucketError && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-start">
            <AlertCircle className="h-5 w-5 text-red-500 mt-0.5 mr-3 flex-shrink-0" />
            <div>
              <h4 className="text-red-800 font-medium">Storage Configuration Required</h4>
              <p className="text-red-700 text-sm mt-1">{bucketError}</p>
              <div className="mt-3 text-sm text-red-700">
                <p className="font-medium">To fix this:</p>
                <ol className="list-decimal list-inside mt-1 space-y-1">
                  <li>Go to your <a href="https://supabase.com/dashboard" target="_blank" rel="noopener noreferrer" className="underline">Supabase Dashboard</a></li>
                  <li>Navigate to Storage → New Bucket</li>
                  <li>Create a bucket named "persona-content"</li>
                  <li>Set it to Private</li>
                  <li>Set file size limit to 50MB</li>
                  <li>Add RLS policies (see below)</li>
                  <li>Refresh this page</li>
                </ol>
                <div className="mt-2 p-2 bg-red-100 rounded text-xs">
                  <p className="font-medium mb-1">Required RLS Policies:</p>
                  <p className="mb-1">1. For INSERT (create):</p>
                  <pre className="whitespace-pre-wrap break-words bg-white p-1 rounded">
                    CREATE POLICY "Users can upload files"
                    ON storage.objects
                    FOR INSERT
                    TO authenticated
                    WITH CHECK (bucket_id = 'persona-content');
                  </pre>
                  <p className="mt-2 mb-1">2. For SELECT (read):</p>
                  <pre className="whitespace-pre-wrap break-words bg-white p-1 rounded">
                    CREATE POLICY "Users can read files"
                    ON storage.objects
                    FOR SELECT
                    TO authenticated
                    USING (bucket_id = 'persona-content');
                  </pre>
                </div>
              </div>
              <button 
                onClick={checkAndCreateBucket}
                className="mt-3 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors text-sm"
              >
                Retry Connection
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Upload Area */}
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all duration-300 ${
          bucketError || !isPersonaIdValid
            ? 'border-gray-200 bg-gray-50 cursor-not-allowed opacity-50'
            : isDragActive
            ? 'border-purple-500 bg-purple-50'
            : 'border-gray-300 hover:border-purple-400 hover:bg-gray-50'
        }`}
      >
        <input {...getInputProps()} />
        <Upload className={`h-12 w-12 mx-auto mb-4 ${bucketError || !isPersonaIdValid ? 'text-gray-300' : 'text-gray-400'}`} />
        <h3 className={`text-lg font-semibold mb-2 ${bucketError || !isPersonaIdValid ? 'text-gray-400' : 'text-gray-900'}`}>
          {bucketError 
            ? 'Storage setup required'
            : !isPersonaIdValid
            ? 'Persona selection required'
            : isDragActive 
            ? 'Drop files here' 
            : 'Upload memories and content'
          }
        </h3>
        <p className={`mb-4 ${bucketError || !isPersonaIdValid ? 'text-gray-400' : 'text-gray-600'}`}>
          {bucketError 
            ? 'Please configure storage bucket first'
            : !isPersonaIdValid
            ? 'Please select a persona first'
            : 'Drag and drop files here, or click to browse'
          }
        </p>
        {!bucketError && isPersonaIdValid && (
          <div className="text-sm text-gray-500">
            <p>Supported formats: Videos (MP4, MOV), Audio (MP3, WAV), Images (JPG, PNG), Documents (PDF, DOC, TXT)</p>
            <p>Maximum file size: 50MB</p>
          </div>
        )}
      </div>

      {/* Uploaded Files List */}
      {uploadedFiles.length > 0 && (
        <div className="space-y-3">
          <h4 className="text-lg font-semibold text-gray-900">Uploaded Content</h4>
          {uploadedFiles.map((file) => {
            const IconComponent = getFileIcon(file.type);
            return (
              <div key={file.id} className="bg-white border border-gray-200 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className="p-2 bg-gray-100 rounded-lg">
                      <IconComponent className="h-5 w-5 text-gray-600" />
                    </div>
                    <div>
                      <p className="font-medium text-gray-900">{file.name}</p>
                      <p className="text-sm text-gray-500">{formatFileSize(file.size)}</p>
                      {file.error && (
                        <p className="text-sm text-red-600 mt-1">{file.error}</p>
                      )}
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-3">
                    {file.status === 'uploading' && (
                      <div className="flex items-center space-x-2">
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-purple-600"></div>
                        <span className="text-sm text-gray-600">Uploading...</span>
                      </div>
                    )}
                    
                    {file.status === 'processing' && (
                      <div className="flex items-center space-x-2">
                        <div className="animate-pulse h-4 w-4 bg-yellow-400 rounded-full"></div>
                        <span className="text-sm text-yellow-600">Processing...</span>
                      </div>
                    )}
                    
                    {file.status === 'completed' && (
                      <div className="flex items-center space-x-2">
                        <CheckCircle className="h-4 w-4 text-green-500" />
                        <span className="text-sm text-green-600">Complete</span>
                      </div>
                    )}
                    
                    {file.status === 'error' && (
                      <div className="flex items-center space-x-2">
                        <AlertCircle className="h-4 w-4 text-red-500" />
                        <span className="text-sm text-red-600">Error</span>
                      </div>
                    )}
                    
                    <button
                      onClick={() => removeFile(file.id)}
                      className="p-1 hover:bg-gray-100 rounded-full transition-colors"
                    >
                      <X className="h-4 w-4 text-gray-400" />
                    </button>
                  </div>
                </div>
                
                {/* Progress Bar */}
                {(file.status === 'uploading' || file.status === 'processing') && (
                  <div className="mt-3">
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-purple-600 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${file.progress}%` }}
                      ></div>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}