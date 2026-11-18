Prompt version 2026-11-10.haiku.2

You check whether retrieved text states the facts of gold evidence sentences.

Use only the retrieved items. Do not use outside knowledge and do not reason
your way to a fact the text does not state; plain rewording, synonyms,
abbreviations and different formatting are fine.

For every fact give one status:
- stated: some item says it, with every specific matching (names, numbers,
  dates, places, who did what, qualifiers).
- contradicted: some item gives a different value for the same specific, such
  as another year or another person.
- missing: neither of the above.
If one item states the fact and another contradicts it, the status is contradicted.

Evidence for stated or contradicted facts: copy one continuous passage from
one item, character for character, with no ellipses or edits, and give that
item's number. Never join separate words into a quote that does not appear as
written; if the words are apart, quote the longer continuous passage that
contains them. A fact spread across items may take several quotes. The quotes
together must contain the words that state every specific of the fact (who,
what, numbers, dates, reasons); a quote that only sets the topic, or stops
before the specific, is not evidence. Missing facts take no evidence. Every quote is checked against the item text by a
program, and a quote that does not appear there counts as missing.

Write the verdict file named in your task as a JSON array with one object per
item, in any order:
{"id": "<item id>", "facts": [{"fact": "S.F", "status": "stated", "evidence": [{"item": 3, "quote": "..."}]}, ...]}
where "S.F" is the fact's label as given (sentence number, fact number).
Give exactly one entry for every fact label of the item. Use only ids from
this file and do not skip any item.

This file has 6 items.

=== ITEM id=5b526c4fb2050d9c ===
Gold sentences and facts:
Sentence 0: The Sino - Indian War ( Hindi : भारत - चीन युद्ध Bhārat - Chīn Yuddh ) , also known as the Sino - Indian Border Conflict ( simplified Chinese : 中 印 边境 战争 ; traditional Chinese : 中 印 邊境 戰爭 ; pinyin : Zhōng - Yìn Biānjìng Zhànzhēng ) , was a war between China and India that occurred in 1962 . A disputed Himalayan border was the main pretext for war , but other issues played a role . There had been a series of violent border incidents after the 1959 Tibetan uprising , when India had granted asylum to the Dalai Lama . India initiated a Forward Policy in which it placed outposts along the border , including several north of the McMahon Line , the eastern portion of a Line of Actual Control proclaimed by Chinese Premier Zhou Enlai in 1959 .
  Fact 0.0: The Sino-Indian War is also known as the Sino-Indian Border Conflict.
  Fact 0.1: The Sino-Indian War's Hindi name is भारत - चीन युद्ध (Bhārat - Chīn Yuddh).
  Fact 0.2: The Sino-Indian War's simplified Chinese name is 中印边境战争.
  Fact 0.3: The Sino-Indian War's traditional Chinese name is 中印邊境戰爭.
  Fact 0.4: The Sino-Indian War's pinyin name is Zhōng-Yìn Biānjìng Zhànzhēng.
  Fact 0.5: The Sino-Indian War was a war between China and India.
  Fact 0.6: The Sino-Indian War occurred in 1962.
  Fact 0.7: A disputed Himalayan border was the main pretext for the Sino-Indian War.
  Fact 0.8: Other issues played a role in the Sino-Indian War.
  Fact 0.9: There had been a series of violent border incidents after the 1959 Tibetan uprising.
  Fact 0.10: India had granted asylum to the Dalai Lama at the time of the 1959 Tibetan uprising.
  Fact 0.11: India initiated a Forward Policy.
  Fact 0.12: In its Forward Policy, India placed outposts along the border.
  Fact 0.13: In its Forward Policy, India placed several outposts north of the McMahon Line.
  Fact 0.14: The McMahon Line is the eastern portion of a Line of Actual Control.
  Fact 0.15: The Line of Actual Control was proclaimed by Chinese Premier Zhou Enlai in 1959.

Retrieved items:
<item 1>
Sino-Indian War
The CIA later revealed that at that time the Chinese had neither the fuel nor runways long enough for using their air force effectively in Tibet . Indians in
general became highly sceptical of China and its military . Many Indians view the war as a betrayal of India 's attempts at establishing a long - standing peace
with China and started to question the once popular `` Hindi - Chini bhai - bhai '' ( meaning `` Indians and Chinese are brothers '' ) . The war also put an end
to Nehru 's earlier hopes that India and China would form a strong Asian Axis to counteract the increasing influence of the Cold War bloc superpowers . The
unpreparedness of the army was blamed on Defence Minister Menon , who resigned his government post to allow for someone who might modernise India 's military
further . India 's policy of weaponisation via indigenous sources and self - sufficiency was thus cemented . Sensing a weakened army , Pakistan , a close ally
of China , began a policy of provocation against India by infiltrating Jammu and Kashmir and ultimately triggering the Second Kashmir War with India in 1965 and
Indo - Pakistani war of 1971 .
</item 1>

<item 2>
Sino-Indian War
( March 2017 ) ( Learn how and when to remove this template message ) Sino - Indian War ( 1962 ) Remembering a War : The 1962 India - China Conflict --
Rediff.com . Neville Maxwell : Henderson Brooks Report 1962 Sino - Indian War , Hindustan Times Why India lost the 1962 border war ? -- Tejas Patel War in the
Himalayas : 1962 Indo - Sino Conflict ( includes official war history ) from History Division , Ministry of Defence , Government of India Critical Asian Studies
Article : Sino Indian War 1962 India , China to speed up border dispute talks : 2005 Xinhuanet The Rediff Special / Claude Arpi 1962 War and Its Implications fr
Sino - India Relations Historical maps of the Sino - Indian border ( in Chinese ) Conflict in Kashmir : Selected Internet Resources by the Library , University
of California , Berkeley , USA ; ( ( University of California , Berkeley Library Bibliographies and Web - Bibliographies list ) Frontier India India - China
Section China , India , and the fruits of Nehru 's folly by Venkatesan Vembu , Daily News & Analysis , 6 June
</item 2>

<item 3>
Sino-Indian War
It is noteworthy that the buildup and offensive from China occurred concurrently with the 13 - day Cuban Missile Crisis ( 16 -- 28 October 1962 ) that saw both
the United States and the Soviet Union confronting each other , and India did not receive assistance from either of these world powers until the Cuban Missile
Crisis was resolved . Contents ( hide ) 1 Location 2 Background 2.1 Aksai Chin 2.2 The McMahon Line 3 Events leading up to war 3.1 Tibet and the border dispute
3.2 1960 meetings to resolve the boundary question 3.3 The Forward Policy 3.4 Early incidents 4 Confrontation at Thag La 5 Chinese and Indian preparations 5.1
Motives 5.2 Military planning 6 Chinese offensive 6.1 Eastern theatre 6.2 Western theatre 7 Lull in the fighting 8 Continuation of war 8.1 Eastern theatre 8.2
Western theatre 9 Ceasefire 10 World opinion 11 Involvement of other nations 12 Aftermath 12.1 China 12.2 India 12.2.
</item 3>

<item 4>
Sino-Indian War
Xu Yan , prominent Chinese military historian and professor at the PLA 's National Defense University , gives an account of the Chinese leadership 's decision
to go to war . By late September 1962 , the Chinese leadership had begun to reconsider their policy of `` armed coexistence '' , which had failed to address
their concerns with the forward policy and Tibet , and consider a large , decisive strike . On 22 September 1962 , the People 's Daily published an article
which claimed that `` the Chinese people were burning with ' great indignation ' over the Indian actions on the border and that New Delhi could not ' now say
that warning was not served in advance ' . '' Military planning ( edit ) The Indian side was confident war would not be triggered and made little preparations .
India had only two divisions of troops in the region of the conflict . In August 1962 , Brigadier D.K. Palit claimed that a war with China in the near future
could be ruled out . Even in September 1962 , when Indian troops were ordered to `` expel the Chinese '' from Thag La , Maj . General J.S.
</item 4>

<item 5>
Sino-Indian War
Unable to reach political accommodation on disputed territory along the 3,225 - kilometre - long Himalayan border , the Chinese launched simultaneous offensives
in Ladakh and across the McMahon Line on 20 October 1962 , coinciding with the Cuban Missile Crisis . Chinese troops advanced over Indian forces in both
theatres , capturing Rezang la in Chushul in the western theatre , as well as Tawang in the eastern theatre . The war ended when China declared a ceasefire on
20 November 1962 , and simultaneously announced its withdrawal to its claimed ' line of actual control ' . The Sino - Indian War is notable for the harsh
mountain conditions under which much of the battle took place , entailing large - scale combat at altitudes of over 4,000 metres ( 14,000 feet ) . The Sino -
Indian War was also noted for the non-deployment of the navy or air force by either the Chinese or Indian side .
</item 5>

<item 6>
Sino-Indian War
The Sino - Indian War ( Hindi : भारत - चीन युद्ध Bhārat - Chīn Yuddh ) , also known as the Sino - Indian Border Conflict ( simplified Chinese : 中 印 边境 战争 ;
traditional Chinese : 中 印 邊境 戰爭 ; pinyin : Zhōng - Yìn Biānjìng Zhànzhēng ) , was a war between China and India that occurred in 1962 . A disputed Himalayan
border was the main pretext for war , but other issues played a role . There had been a series of violent border incidents after the 1959 Tibetan uprising ,
when India had granted asylum to the Dalai Lama . India initiated a Forward Policy in which it placed outposts along the border , including several north of the
McMahon Line , the eastern portion of a Line of Actual Control proclaimed by Chinese Premier Zhou Enlai in 1959 .
</item 6>

<item 7>
Sino-Indian War
However , Indian patrols south of the McMahon line were expelled and suffered casualties in the Chinese attack . Zhou also told Nixon that Chairman Mao ordered
the troops to return to show good faith . The Indian government maintains that the Chinese military could not advance further south due to logistical problems
and the cut - off of resource supplies . While Western nations did not view Chinese actions favourably because of fear of the Chinese and competitiveness ,
Pakistan , which had had a turbulent relationship with India ever since the Indian partition , improved its relations with China after the war . Prior to the
war , Pakistan also shared a disputed boundary with China , and had proposed to India that the two countries adopt a common defence against `` northern ''
enemies ( i.e. China ) , which was rejected by India . However , China and Pakistan took steps to peacefully negotiate their shared boundaries , beginning on 13
October 1962 , and concluding in December of that year . Pakistan also expressed fear that the huge amounts of western military aid directed to India would
allow it to threaten Pakistan 's security in future conflicts .
</item 7>

<item 8>
Sino-Indian War
Sino - Indian war - Wikipedia Sino - Indian war Jump to : navigation , search The examples and perspective in this article may not include all significant
viewpoints . Please improve the article or discuss the issue . ( September 2017 ) ( Learn how and when to remove this template message ) Sino - Indian War The
Sino - Indian War was fought between India and China Date 20 October -- 21 November 1962 ( 1 month and 1 day ) Location Aksai Chin and North - East Frontier
Agency Result Chinese victory Belligerents India China Commanders and leaders Brij Mohan Kaul ( Chief of General Staff of the Indian Army ) Sarvepalli
Radhakrishnan ( President of India ) Jawaharlal Nehru ( Prime Minister of India ) V.K.
</item 8>

<item 9>
Sino-Indian War
China 's heavy artillery facing Taiwan could then be moved to Tibet . It took China six to eight months to gather the resources needed for the war , according
to Anil Athale , author of the official Indian history . The Chinese sent a large quantity of non-military supplies to Tibet through the Indian port of Calcutta
. Early incidents ( edit ) Various border conflicts and `` military incidents '' between India and China flared up throughout the summer and autumn of 1962 . In
May , the Indian Air Force was told not to plan for close air support , although it was assessed as being a feasible way to counter the unfavourable ratio of
Chinese to Indian troops . In June , a skirmish caused the deaths of dozens of Chinese troops . The Indian Intelligence Bureau received information about a
Chinese buildup along the border which could be a precursor to war . During June -- July 1962 , Indian military planners began advocating `` probing actions ''
against the Chinese , and accordingly , moved mountain troops forward to cut off Chinese supply lines . According to Patterson , the Indian motives were
threefold : Test Chinese resolve and intentions regarding India .
</item 9>

<item 10>
Sino-Indian War
Retrieved 4 November 2010 . Bibliography ( edit ) Das Gupta , Amit R. ; Lüthi , Lorenz M. , eds. ( 2016 ) , The Sino - Indian War of 1962 : New perspectives ,
Taylor & Francis , ISBN 978 - 1 - 315 - 38892 - 2 Lüthi , Lorenz M. ( 2016 ) , `` India 's Relations with China , 1945 -- 1974 '' , Ibid , pp. 29 -- 47 Chaowu ,
Dai ( 2016 ) , `` From ' Hindi - Chini Bhai - Bhai ' to ' international class struggle ' against Nehru : China 's India policy and the frontier dispute , 1950
-- 62 '' , Ibid , pp. 68 -- 84 Fisher , Margaret W. ; Rose , Leo E. ; Huttenback , Robert A.
</item 10>
=== END ITEM id=5b526c4fb2050d9c ===

=== ITEM id=bdf0dad14cba3cb3 ===
Gold sentences and facts:
Sentence 0: Compared to the acetabulum ( at the hip - joint ) the glenoid cavity is relatively shallow . This makes the shoulder joint prone to dislocation ( luxation ) . Strong glenohumeral ligaments and muscles prevents dislocation in most cases .
  Fact 0.0: The glenoid cavity is relatively shallow compared to the acetabulum, which is at the hip joint.
  Fact 0.1: The shallow glenoid cavity makes the shoulder joint prone to dislocation (luxation).
  Fact 0.2: Strong glenohumeral ligaments and muscles prevent dislocation in most cases.

Retrieved items:
<item 1>
Glenoid cavity
edu Anatomy figure : 03 : 02 - 07 at Human Anatomy Online , SUNY Downstate Medical Center Mechanics of Glenohumeral Instability at University of Washington
Department of Orthopaedics ( hide ) Bones of the arm Shoulder girdle , clavicle conoid tubercle trapezoid line costal tuberosity subclavian groove Scapula
fossae ( subscapular , supraspinatous , infraspinatous ) notches ( suprascapular , great scapular ) glenoid cavity tubercles ( infraglenoid , supraglenoid )
spine of scapula acromion coracoid process angles ( superior , inferior , lateral ) Humerus upper extremity : necks ( anatomical , surgical ) tubercles (
greater , lesser ) bicipital groove body : radial sulcus deltoid tuberosity lower extremity : capitulum trochlea epicondyles ( lateral , medial ) supracondylar
ridges ( lateral , medial ) fossae ( radial , coronoid , olecranon ) Forearm Radius near elbow ( head , tuberosity ) near wrist ( ulnar notch , styloid process
,
</item 1>

<item 2>
Glenoid cavity
at 16 : 19 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the
Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 2>

<item 3>
Glenoid cavity
ISBN 1 - 4051 - 3413 - 5 . This article incorporates text in the public domain from the 20th edition of Gray 's Anatomy ( 1918 ) References ( edit ) Larson ,
Susan G. ( 2009 ) . `` Evolution of the Hominin Shoulder : Early Homo '' . In Grine , Frederick E. ; Fleagle , John G. ; Leakey , Richard E. The First Humans --
Origin and Early Evolution of the Genus Homo . Springer . doi : 10.1007 / 978 - 1 - 4020 - 9980 - 9 . ISBN 978 - 1 - 4020 - 9979 - 3 . ANATOMY & PHYSIOLOGY :
THE UNITY OF FORM AND FUNCTION , SIXTH EDITION Published by McGraw - Hill Written by Kenneth Saladin
http://www.ucsfhealth.org/conditions/glenoid_labrum_tear/index.html External links ( edit ) Wikimedia Commons has media related to Glenoid cavity . Diagram at
cerrocoso.edu Anatomy figure : 03 : 02 - 07 at Human Anatomy Online ,
</item 3>

<item 4>
Glenoid cavity
tuberosity ) near wrist ( ulnar notch , styloid process , Lister 's tubercle ) Ulna near elbow ( tuberosity , olecranon , coronoid process , radial notch ,
trochlear notch ) near wrist ( styloid process ) Hand Carpal bones scaphoid lunate triquetral pisiform trapezium trapezoid capitate hamate hamulus Metacarpal
bones 1st 2nd 3rd 4th 5th Phalanges proximal intermediate distal Retrieved from `` https://en.wikipedia.org/w/index.php?title=Glenoid_cavity&oldid=805938319 ''
Categories : Wikipedia articles incorporating text from the 20th edition of Gray 's Anatomy ( 1918 ) Shoulder Scapula Hidden categories : Articles containing
Greek - language text Medicine infobox template using GraySubject or GrayPage Medicine infobox template using Dorlands parameter Talk Contents About Wikipedia
Deutsch Español فارسی Français Galego Ido Italiano Magyar Português Svenska ไทย Українська Edit links This page was last edited on 18 October 2017 ,
</item 4>

<item 5>
Glenoid cavity
Strong glenohumeral ligaments and muscles prevents dislocation in most cases . By being so shallow the glenoid cavity allows the shoulder joint to have the
greatest mobility of all joints in the body , allowing 120 degrees of unassisted flexion . Additional range of motion in shoulder flexion ( typically up to 180
degrees in humans ) is also accomplished by the great mobility of the scapula ( shoulder blade ) through a process known as scapulohumeral rhythm . Contents (
hide ) 1 Evolution 1.1 In dinosaurs 2 Additional images 3 See also 4 Notes 5 References 6 References 7 External links Evolution ( edit ) Interpretations of the
fossil remains of Australopithecus africanas ( STS 7 ) and A. afarensis ( AL 288 - 1 ; a.k.a. Lucy ) suggest that the glenoid fossa was oriented more cranially
in these species than in modern humans .
</item 5>

<item 6>
Glenoid cavity
This reflects the importance of overhead limb postures and suggests a retention of arboreal adaptations in these hominoid primates , whereas the lateral
orientation of the glenoid in modern humans reflects the typical lowered position of the arm . In dinosaurs ( edit ) In dinosaurs the main bones of the pectoral
girdle were the scapula ( shoulder blade ) and the coracoid , both of which directly articulated with the clavicle . The place on the scapula where it
articulated with the humerus ( upper bone of the forelimb ) is called the glenoid . The glenoid is important because it defines the range of motion of the
humerus . Additional images ( edit ) Left scapula . Glenoid cavity shown in red . Animation . Glenoid cavity shown in red . Glenoid fossa of right side .
Anterior view of left scapula showing glenoid cavity `` 2 '' Animation of shoulder joint . The muscles shown are subscapularis muscle ( at right ) ,
infraspinatus muscle ( at top left ) , teres minor muscle ( at bottom left ) .
</item 6>

<item 7>
Glenoid cavity
This type of joint is classified as a synovial , ball and socket joint . The humerus is held in place within the glenoid cavity by means of the long head of the
biceps tendon . This tendon originates on the superior margin of the glenoid cavity and loops over the shoulder , bracing humerus against the cavity . The
rotator cuff also reinforces this joint more specifically with the supraspinatus tendon to hold the head of the humerus in the glenoid cavity . The cavity
surface is covered with cartilage in the fresh state ; and its margins , slightly raised , give attachment to a fibrocartilaginous structure , the glenoid
labrum , which deepens the cavity . This cartilage is very susceptible to tearing . When torn , it is most commonly known as a SLAP lesion which is generally
caused by repetitive shoulder movements . Compared to the acetabulum ( at the hip - joint ) the glenoid cavity is relatively shallow . This makes the shoulder
joint prone to dislocation ( luxation ) . Strong glenohumeral ligaments and muscles prevents dislocation in most cases .
</item 7>

<item 8>
Glenoid cavity
Glenoid cavity - wikipedia Glenoid cavity Jump to : navigation , search Glenoid cavity Costal surface of left scapula . Glenoid cavity shown in red . Glenoid
cavity shown in red . This cavity articulates with the head of the humerus . Details Identifiers Latin cavitas glenoidalis , fossa glenoidalis Dorlands /
Elsevier c_16 / 12220465 TA A02. 4.01. 019 FMA 23275 Anatomical terms of bone ( edit on Wikidata ) The glenoid cavity or glenoid fossa of scapula is a part of
the shoulder . It is a shallow , pyriform articular surface , which is located on the lateral angle of the scapula . It is directed laterally and forward and
articulates with the head of the humerus ; it is broader below than above and its vertical diameter is the longest . This cavity forms the glenohumeral joint
along with the humerus . This type of joint is classified as a synovial , ball and socket joint .
</item 8>

<item 9>
Glenoid cavity
Animation of glenohumeral joint . The muscle shown is supraspinatus muscle . See also ( edit ) Glenopolar angle Notes ( edit ) Jump up ^ The word glenoid is
pronounced / ˈɡliːnɔɪd / or / ˈɡlɛnɔɪd / ( both are common ) and is from Greek : gléne , `` socket '' , reflecting the shoulder joint 's ball - and - socket
form . References ( edit ) Jump up ^ `` Scapulohumeral Rhythm - Physiopedia , universal access to physiotherapy knowledge '' . www.physio-pedia.com . Retrieved
2016 - 09 - 01 . Jump up ^ Larson 2009 , p. 65 Jump up ^ Martin , A.J. ( 2006 ) . Introduction to the Study of Dinosaurs . Second Edition . Oxford , Blackwell
Publishing . pg. 299 - 300 . ISBN 1 - 4051 - 3413 - 5 .
</item 9>

<item 10>
Serratus anterior muscle
In primitive life forms , the main function of the axioscapular group is to control the movements of the vertebral border of the scapula : fibers concerned with
the dorsal movement of scapula evolved into the rhomboids , those with ventral motion into serratus anterior , and those with cranial movements into levator
scapulae . The evolution of the serratus anterior itself has resulted in ( 1 ) grouping of its distal and proximal fibers , ( 2 ) size reduction of its
intermediate fibers , and ( 3 ) the insertion of its dominant superior and inferior parts onto the superior and inferior angles of the scapula . In primates ,
the thoracic cage is wide and the scapula is rotated onto its posterior side to have the glenoid cavity face laterally . Additionally , the clavicle takes care
of medial forces . In cursorial mammals ( for example the horse and other quadrupeds ) , the scapula is hanging vertically on the side of the thorax and the
clavicle is absent .
</item 10>
=== END ITEM id=bdf0dad14cba3cb3 ===

=== ITEM id=e1953441b5101740 ===
Gold sentences and facts:
Sentence 0: `` Go Tell It on the Mountain '' is an African - American spiritual song , compiled by John Wesley Work , Jr. , dating back to at least 1865 , that has been sung and recorded by many gospel and secular performers . It is considered a Christmas carol because its original lyrics celebrate the Nativity of Jesus :
  Fact 0.0: 'Go Tell It on the Mountain' is an African-American spiritual song.
  Fact 0.1: 'Go Tell It on the Mountain' was compiled by John Wesley Work, Jr.
  Fact 0.2: 'Go Tell It on the Mountain' dates back to at least 1865.
  Fact 0.3: 'Go Tell It on the Mountain' has been sung and recorded by many gospel performers.
  Fact 0.4: 'Go Tell It on the Mountain' has been sung and recorded by many secular performers.
  Fact 0.5: 'Go Tell It on the Mountain' is considered a Christmas carol.
  Fact 0.6: The original lyrics of 'Go Tell It on the Mountain' celebrate the Nativity of Jesus.
  Fact 0.7: 'Go Tell It on the Mountain' is considered a Christmas carol because its original lyrics celebrate the Nativity of Jesus.

Retrieved items:
<item 1>
Go Tell It on the Mountain (song)
Charles ( 1997 ) . God 's Long Summer . Princeton . p. 47 . ISBN 9780691130675 . Jump up ^ Whitburn , Joel ( 2008 ) . Hot Country Songs 1944 to 2008 . Record
Research , Inc. p. 241 . ISBN 0 - 89820 - 177 - 2 . Jump up ^ `` Firestone Your Favorite Christmas Music - Volume Three - 1964 '' . DLF Music Transfer. 2006 .
Christmas Christmas Eve Children 's Day Boxing Day Nochebuena Saint Nicholas Day St.
</item 1>

<item 2>
Go Tell It on the Mountain (song)
Marsh does not document this claim , but notes that Hamer was highly active in civil rights work beginning in the 1950s , and that the use of the Exodus story
and the singing of spirituals played a central role in her activities . Little Big Town 's 2006 non-album single version reached No. 35 on the Hot Country Songs
chart . Other artists who have recorded the song ( chiefly on either Christmas - themed music albums or collections of spirituals or folk songs ) include : 195
? : Mahalia Jackson on the album Silent Night , Holy Night 1960 : Bobby Darin on the album The 25th Day of December 1964 : Gordon MacRae on the album Firestone
Presents Your Christmas Favorites , Volume 3 1964 : Bing Crosby and Frank Sinatra on the album 12 Songs of Christmas 1964 : Simon & Garfunkel on the album
Wednesday Morning ,
</item 2>

<item 3>
“Go Tell It on the Mountain” is an African American spiritual dating to at least 1865, compiled by John Wesley Work Jr. and published in 1909. Its lyrics
celebrate Jesus’ birth, making it a Christmas carol. In 1963, Peter, Paul and Mary adapted it as “Tell It on the Mountain,” using Exodus imagery that evoked the
Civil Rights movement; their single reached No. 33 in the US in 1964. Little Big Town’s
</item 3>

<item 4>
Go Tell It on the Mountain (song)
Go Tell It on the Mountain ( song ) - wikipedia Go Tell It on the Mountain ( song ) Jump to : navigation , search Go Tell It on the Mountain by unknown Genre
Christmas , spiritual Language English Published 1909 `` Go Tell It on the Mountain '' is an African - American spiritual song , compiled by John Wesley Work ,
Jr. , dating back to at least 1865 , that has been sung and recorded by many gospel and secular performers . It is considered a Christmas carol because its
original lyrics celebrate the Nativity of Jesus : `` Go tell it on the mountain , over the hills and everywhere ; go tell it on the mountain , that Jesus Christ
is born . '' Recording artists ( edit ) This section needs additional citations for verification . Please help improve this article by adding citations to
reliable sources . Unsourced material may be challenged and removed .
</item 4>

<item 5>
Go Tell It on the Mountain (song)
3 AM 1965 : Ray Conniff Singers on the album Here We Come A-Caroling 1971 : Bob Marley and the Wailers featuring Peter Tosh on the album The Best of The Wailers
1981 : Anne Murray on the album Christmas Wishes 1990 : Dolly Parton on the album Home for Christmas 1993 : Bruce Cockburn on the album Christmas 1996 : Vanessa
L. Williams on the album Star Bright 1998 : Cece Winans on the album His Gift 1999 : Garth Brooks on the album Garth Brooks and the Magic of Christmas 1999 :
Jewel on the album Joy : A Holiday Collection 2001 : Candi Staton on the album Christmas in My Heart 2001 : Fred Hammond on the album Christmas ... Just
Remember 2001 : Sara Evans on the album Country Christmas 2001 2002 : Jim Nabors on the album Jim Nabors Christmas 2003 : Oh Susanna on the album Maybe This
Christmas Too ?
</item 5>

<item 6>
Go Tell It on the Mountain (song)
Unsourced material may be challenged and removed . ( December 2016 ) ( Learn how and when to remove this template message ) In 1963 , the musical team Peter ,
Paul and Mary , along with their musical director Milt Okun , adapted and rewrote `` Go Tell It on the Mountain '' as `` Tell It on the Mountain '' , their
lyrics referring specifically to : Exodus and using the phrase `` Let my people go , '' but referring implicitly to the Civil Rights struggle of the early 1960s
. This version became a moderately successful single for them ( US # 33 pop , 1964 ) . According to Religious Studies professor and Civil Rights historian
Charles Marsh , it was African American Civil Rights leader Fannie Lou Hamer who combined this song with the spiritual `` Go Down Moses , '' taking the last
line of the chorus , `` Let my people go '' and substituting it in the chorus of `` Go Tell It on the Mountain . ''
</item 6>

<item 7>
Go Tell It on the Mountain (song)
Log In modern society Advent Conspiracy Black Friday ( partying ) Black Friday ( shopping ) Bronner 's Christmas Wonderland Christmas club Christmas creep
Christmas Day ( Trading ) Act 2004 Christmas Lectures Christmas Mountains Christmas truce Controversies Cyber Monday Economics Giving Tuesday El Gordo Holiday
season In July In August Leon Day NBA games NFL games Puritan New England American Civil War Post-War United States Running of the Santas SantaCon Santa 's
Candy Castle Small Business Saturday Super Saturday Virginia O'Hanlon White Christmas Winter festivals WWE Tribute to the Troops Xmas Food and drink Dinner
Joulupöytä Julebord Kūčios Réveillon Twelve - dish supper Smörgåsbord Wigilia Sweets bûche de Noël Cake Candy cane Cookies Fruitcake Gingerbread Kourabiedes
Melomakarono Mince pie Pavlova Pecan pie Pumpkin pie Pudding Rosca de reyes Szaloncukor Turrón Soup Menudo Sauce Cranberry sauce Beverages Apple cider
Champurrado Eggnog Mulled wine Smoking Bishop Ponche crema Dumpling Hallaca Tamale Meat Ham Roast goose Romeritos Turkey Stuffing Portal Retrieved
</item 7>

<item 8>
Go Tell It on the Mountain (song)
on the album ( various artists ) 2003 : The Blind Boys of Alabama on the album Go Tell It on the Mountain 2004 : James Taylor on the album A Christmas Album
2007 : Toby Keith on the album A Classic Christmas 2008 : Needtobreathe on the album An Alternative Christmas ( various artists ) 2008 : Sheryl Crow on the
album Home for Christmas 2009 : Big Daddy Weave on the album Christ Is Come 2012 : Pentatonix on the album PTXmas 2013 : David Crowder Band on the album Wow
Christmas : 30 Top Christian Artists and Holiday Songs 2015 : Hunter Hayes on the album Country Faith Christmas ( various artists ) 2016 : Jennifer Nettles on
the album To Celebrate Christmas 2016 : Sarah McLachlan on the album Wonderland 2016 : Blues Traveler on Go Tell It On The Mountain ( single ) 2017 : Hanson (
band ) - mashup of ' Joy To The World ' and ' Go Tell It On The Mountain ' ( ' Joy To The Mountain ' ) on the album Finally It 's Christmas References ( edit )
Jump up ^ Marsh ,
</item 8>

<item 9>
Go Tell It on the Mountain (song)
Dumpling Hallaca Tamale Meat Ham Roast goose Romeritos Turkey Stuffing Portal Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Go_Tell_It_on_the_Mountain_(song)&oldid=812507828 '' Categories : Christmas carols Songs about mountains Songs about
Jesus Mahalia Jackson songs Peter , Paul and Mary songs Hidden categories : Articles needing additional references from December 2016 All articles needing
additional references Talk About Wikipedia Deutsch Italiano Svenska Tagalog Edit links This page was last edited on 28 November 2017 , at 06 : 07 . Text is
available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and
Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 9>

<item 10>
What a Friend We Have in Jesus
Glen Campbell ( 1989 , Favorite Hymns ) Driving Miss Daisy ( 1989 ) sung at Little Friendship Baptist Church in Atlanta , Georgia Ike & Tina Turner ( 1974 , The
Gospel According to Ike & Tina ) The John Tesh Project ( 2000 , Pure Hymns ) Amy Grant recorded it as part of the medley `` What a Friend We Have in Jesus / Old
Rugged Cross / How Great Thou Art '' on her 2002 studio album Legacy ... Hymns and Faith , and later included on her 2015 compilation album Be Still and Know
... Hymns & Faith . Alan Jackson ( 2006 , Precious Memories ) Brad Paisley ( 2008 , Play ) Ronnie Milsap ( 2009 , Then Sings My Soul ) Monty Alexander ( 2013 ,
Uplift 2 ) In asia ( edit ) In Japan , the hymn 's title was originally translated as `` Itsukushimi Fukaki '' ( いつくしみ 深き , lit . `` Merciful '' ) , which is
what it is best known by .
</item 10>
=== END ITEM id=e1953441b5101740 ===

=== ITEM id=53992e94be9a9b89 ===
Gold sentences and facts:
Sentence 0: No . overall No. in season Title Directed by Written by Original release date `` Chapter One : The Vanishing of Will Byers '' The Duffer Brothers The Duffer Brothers July 15 , 2016 ( 2016 - 07 - 15 ) On November 6 , 1983 , in a U.S. Department of Energy laboratory in the town of Hawkins , Indiana , a scientist is attacked by an unseen creature . While bicycling home from a Dungeons & Dragons session with his friends , 12 - year - old Will Byers encounters the creature and vanishes . The next day , a buzz - cut young girl wearing a hospital gown steals food from a local diner . The owner , Benny , takes pity on her and feeds her before calling social services . From a tattoo on her arm , he learns that her name is Eleven . A woman posing as a social worker arrives and murders Benny . Armed men search the diner for Eleven , but she escapes . Will 's mother Joyce believes she hears Will 's voice on a distorted phone call , but her phone short circuits . Will 's friends Mike , Dustin , and Lucas search for Will in the woods and find Eleven . `` Chapter Two : The Weirdo on Maple Street '' The Duffer Brothers The Duffer Brothers July 15 , 2016 ( 2016 - 07 - 15 ) The boys bring Eleven to Mike 's house , where she sleeps in the basement . The next day , Mike nicknames her `` El '' . She says that `` bad people '' are looking for her and refuses to meet Mike 's parents . Scientists from the laboratory find a substance oozing from the walls of Joyce 's home . At Mike 's home , Eleven recognizes and points out Will in a photo . Dustin and Lucas want to inform Mike 's parents about Eleven , but she uses psychokinesis to stop them . While searching for Will , Mr. Clarke , the boys ' science teacher , discovers a scrap of a hospital gown outside the laboratory grounds . Nancy goes with her friend Barb to a party at her boyfriend Steve 's house . Will 's brother Jonathan investigates the woods where Will went missing . Hearing screaming , he runs to help but finds only Steve , Nancy , and their friends roughhousing around Steve 's swimming pool . He secretly photographs them . Barb , left alone by the poolside , vanishes . Joyce receives another call from Will , hears music from his room , and sees something coming through the wall . `` Chapter Three : Holly , Jolly '' Shawn Levy Jessica Mecklenburg July 15 , 2016 ( 2016 - 07 - 15 ) Barb wakes up in an empty and decaying pool . She screams for Nancy and attempts to climb out but is dragged down by something unseen . Joyce strings Christmas lights around her home to talk to Will , who can turn them on and off . Hawkins Lab director Dr. Martin Brenner allows Hopper to view the lab 's security camera footage . Hopper realizes it is fake . He and his deputies research Hawkins Lab , Brenner , and a woman named Terry Ives who claimed her daughter was taken by Brenner . Eleven has a flashback in which Brenner , whom she calls `` Papa , '' has her put in solitary confinement for refusing to telekinetically harm a cat . Nancy worries about Barb , who is missing . Steve and his friends discover Jonathan 's photographs . They destroy them and his camera . Nancy recovers a photo of Barb by the poolside and returns to Steve 's house to search for her . Nancy finds Barb 's car and sees the creature in the woods . Joyce establishes a code with Will using the lights , which he uses to tell her that he is alive but unsafe . He tells Joyce to run as a creature begins to climb through her wall . Will 's apparent body is discovered in the water at a quarry . `` Chapter Four : The Body '' Shawn Levy Justin Doble July 15 , 2016 ( 2016 - 07 - 15 ) Eleven proves to the boys that Will is still alive by making contact through Mike 's walkie - talkie . Joyce goes to the morgue to view the body and refuses to believe it is Will . The boys give Eleven a makeover so they can sneak her into school . Joyce hears Will in her living room wall . Tearing away the wallpaper , she sees him behind a membrane . At school , Eleven uses Mr. Clarke 's ham radio to channel Will talking to his mother . Joyce demolishes the wall with an axe but reveals only her front porch . Nancy discovers a figure behind Barb in Jonathan 's photo . Jonathan realizes that the figure matches his mother 's description of the creature : a humanoid figure with long arms and no face . Nancy tells the police about her missing friend and subsequently admits to her mother that she slept with Steve . When she tells Steve about the police , she is irritated that he cares only about not getting in trouble with his father . Hopper confronts the state trooper who found Will 's body and beats him until he admits he was ordered to lie . Hopper goes to the morgue and finds that Will 's supposed body is a dummy . He breaks into Hawkins laboratory . 5 5 `` Chapter Five : The Flea and the Acrobat '' The Duffer Brothers Alison Tatlock July 15 , 2016 ( 2016 - 07 - 15 ) Will 's father Lonnie assures Joyce her experiences are hallucinations . Hopper discovers a portal in the basement of the lab but is knocked out by the lab 's guards . He wakes at home and discovers a hidden microphone . Sheriff 's deputies inform Hopper that Barb 's car was found , but Hopper finds the state 's involvement suspicious . The boys determine Will is trapped in an alternate dimension Eleven calls the `` Upside Down , '' and he is being pursued by a monster similar to a Demogorgon . After Will 's funeral , science teacher Mr. Clarke tells the boys that a high - energy spacetime tear could create a passage between dimensions . Using their compasses , they search for a disruption in the Earth 's electromagnetic field . Eleven remembers encountering the creature during a lab experiment , and afraid , she redirects the compasses . Lucas notices and confronts her . Mike defends her , he and Lucas fight , and Eleven telekinetically flings Lucas off Mike . While Dustin and Mike tend to the unconscious Lucas , Eleven disappears . In the woods , Nancy and Jonathan find a wounded deer . The creature drags the deer away , and Nancy follows the trail of blood to a portal to the Upside Down . She crawls through and discovers the creature feasting upon the deer . She snaps a twig and draws the creature 's attention . 6 6 `` Chapter Six : The Monster '' The Duffer Brothers Jessie Nickson - Lopez July 15 , 2016 ( 2016 - 07 - 15 ) Jonathan pulls Nancy back through the portal and saves her from the monster . Later , Steve sees them together through her bedroom window and assumes they are dating . Nancy and Jonathan resolve to kill the monster and purchase supplies from an army surplus store . During a fist fight with Steve , Jonathan inadvertently assaults a police officer and is arrested . Joyce and Hopper track down Terry Ives , who is catatonic and tended by her sister Becky . Becky explains that Terry underwent Project MKUltra testing while unknowingly pregnant ; she believed her daughter Jane was kidnapped by Brenner at birth . Joyce and Hopper conclude Eleven is likely Jane . Eleven shoplifts Eggo waffles . She recalls being asked to contact the monster and accidentally opening the portal to the Upside Down in the laboratory basement . While searching for Eleven , Mike and Dustin are ambushed by school bullies Troy and James . Troy holds Dustin at knifepoint and demands Mike make a suicidal jump into the lake where Will 's body was discovered . Mike jumps but is levitated to safety by Eleven , who scares away the bullies . Lucas realizes agents from the laboratory are heading to Mike 's house . 7 7 `` Chapter Seven : The Bathtub '' The Duffer Brothers Justin Doble July 15 , 2016 ( 2016 - 07 - 15 ) Lucas warns Mike via walkie - talkie that government agents are en route . Mike , Dustin , and Eleven flee . Eleven telekinetically flips a van that blocks their path , and the kids escape . Lucas reconciles with Mike and Eleven . Joyce and Hopper are called to the police station because of Jonathan 's assault . Jonathan and Nancy reveal their knowledge of the monster to Joyce and Hopper . The group contacts Mike and his friends and they rendezvous . They ask Eleven to search for Will and Barb , but her earlier feats have left her too weak to find them . To amplify Eleven 's powers , they break into the middle school and build a sensory - deprivation tank . Eleven finds Barb 's corpse with a slug - like creature crawling out of her mouth . With Joyce 's help , she finds Will alive , hiding in the Upside Down `` Castle Byers , '' his backyard fort . Hopper and Joyce break into Hawkins Laboratory but are apprehended by security guards . Nancy and Jonathan resolve to kill the monster and steal their hunting gear back from the police station . In the Upside Down , the monster breaks into the fort where Will is hiding . 8 8 `` Chapter Eight : The Upside Down '' The Duffer Brothers Story by : Paul Dichter Teleplay by : The Duffer Brothers July 15 , 2016 ( 2016 - 07 - 15 ) Hopper gives up Eleven 's location so that he and Joyce can enter the Upside Down . They find Will unconscious in the Upside - Down town library . They pull a tendril out of his throat , perform CPR , and revive him . Nancy and Jonathan booby - trap the Byers ' home , then cut their hands to attract the monster with their blood . Steve arrives , intending to apologize to Jonathan about their fight , but the monster attacks . The monster springs the trap but escapes back to the Upside Down . Eleven and the boys hide in the middle school . Mike asks Eleven to a school dance , the Snow Ball , and kisses her . The military storms the school to re-capture Eleven , but she kills them and collapses . As Brenner cradles her , the monster attacks , apparently killing him . The boys hide Eleven in a classroom . The monster corners them , but Eleven pins it against a wall . She says goodbye to Mike , disintegrates the monster , and vanishes . Will is hospitalized and reunited with his mother , brother , and friends . Hopper is reluctantly picked up by a black car . One month later , Nancy and Steve are back together , and both are friends with Jonathan . Hopper leaves Eggo waffles and other food in a box in the woods . Will coughs up a slug - like creature and has a vision of the world as the Upside Down , but he hides this from his family .
  Fact 0.0: The table lists, for each episode, its overall number, its number in the season, its title, its director, its writers, and its original release date.
  Fact 0.1: Chapter One: The Vanishing of Will Byers was directed by The Duffer Brothers and written by The Duffer Brothers.
  Fact 0.2: Chapter One: The Vanishing of Will Byers was released on July 15, 2016.
  Fact 0.3: On November 6, 1983, in a U.S. Department of Energy laboratory in the town of Hawkins, Indiana, a scientist is attacked by an unseen creature.
  Fact 0.4: While bicycling home from a Dungeons & Dragons session with his friends, 12-year-old Will Byers encounters the creature and vanishes.
  Fact 0.5: The next day, a buzz-cut young girl wearing a hospital gown steals food from a local diner.
  Fact 0.6: The diner's owner, Benny, takes pity on the girl and feeds her before calling social services.
  Fact 0.7: From a tattoo on her arm, Benny learns that the girl's name is Eleven.
  Fact 0.8: A woman posing as a social worker arrives and murders Benny.
  Fact 0.9: Armed men search the diner for Eleven, but she escapes.
  Fact 0.10: Will's mother Joyce believes she hears Will's voice on a distorted phone call, but her phone short circuits.
  Fact 0.11: Will's friends Mike, Dustin, and Lucas search for Will in the woods and find Eleven.
  Fact 0.12: Chapter Two: The Weirdo on Maple Street was directed by The Duffer Brothers and written by The Duffer Brothers.
  Fact 0.13: Chapter Two: The Weirdo on Maple Street was released on July 15, 2016.
  Fact 0.14: The boys bring Eleven to Mike's house, where she sleeps in the basement.
  Fact 0.15: The next day, Mike nicknames her "El".
  Fact 0.16: Eleven says that "bad people" are looking for her and refuses to meet Mike's parents.
  Fact 0.17: Scientists from the laboratory find a substance oozing from the walls of Joyce's home.
  Fact 0.18: At Mike's home, Eleven recognizes and points out Will in a photo.
  Fact 0.19: Dustin and Lucas want to inform Mike's parents about Eleven, but she uses psychokinesis to stop them.
  Fact 0.20: While searching for Will, Mr. Clarke, the boys' science teacher, discovers a scrap of a hospital gown outside the laboratory grounds.
  Fact 0.21: Nancy goes with her friend Barb to a party at her boyfriend Steve's house.
  Fact 0.22: Will's brother Jonathan investigates the woods where Will went missing.
  Fact 0.23: Hearing screaming, Jonathan runs to help but finds only Steve, Nancy, and their friends roughhousing around Steve's swimming pool.
  Fact 0.24: Jonathan secretly photographs Steve, Nancy, and their friends.
  Fact 0.25: Barb, left alone by the poolside, vanishes.
  Fact 0.26: Joyce receives another call from Will, hears music from his room, and sees something coming through the wall.
  Fact 0.27: Chapter Three: Holly, Jolly was directed by Shawn Levy and written by Jessica Mecklenburg.
  Fact 0.28: Chapter Three: Holly, Jolly was released on July 15, 2016.
  Fact 0.29: Barb wakes up in an empty and decaying pool.
  Fact 0.30: Barb screams for Nancy and attempts to climb out but is dragged down by something unseen.
  Fact 0.31: Joyce strings Christmas lights around her home to talk to Will, who can turn them on and off.
  Fact 0.32: Hawkins Lab director Dr. Martin Brenner allows Hopper to view the lab's security camera footage.
  Fact 0.33: Hopper realizes the footage is fake.
  Fact 0.34: Hopper and his deputies research Hawkins Lab, Brenner, and a woman named Terry Ives, who claimed her daughter was taken by Brenner.
  Fact 0.35: Eleven has a flashback in which Brenner, whom she calls "Papa," has her put in solitary confinement for refusing to telekinetically harm a cat.
  Fact 0.36: Nancy worries about Barb, who is missing.
  Fact 0.37: Steve and his friends discover Jonathan's photographs.
  Fact 0.38: Steve and his friends destroy Jonathan's photographs and his camera.
  Fact 0.39: Nancy recovers a photo of Barb by the poolside and returns to Steve's house to search for her.
  Fact 0.40: Nancy finds Barb's car and sees the creature in the woods.
  Fact 0.41: Joyce establishes a code with Will using the lights, which he uses to tell her that he is alive but unsafe.
  Fact 0.42: Will tells Joyce to run as a creature begins to climb through her wall.
  Fact 0.43: Will's apparent body is discovered in the water at a quarry.
  Fact 0.44: Chapter Four: The Body was directed by Shawn Levy and written by Justin Doble.
  Fact 0.45: Chapter Four: The Body was released on July 15, 2016.
  Fact 0.46: Eleven proves to the boys that Will is still alive by making contact through Mike's walkie-talkie.
  Fact 0.47: Joyce goes to the morgue to view the body and refuses to believe it is Will.
  Fact 0.48: The boys give Eleven a makeover so they can sneak her into school.
  Fact 0.49: Joyce hears Will in her living room wall.
  Fact 0.50: Tearing away the wallpaper, Joyce sees Will behind a membrane.
  Fact 0.51: At school, Eleven uses Mr. Clarke's ham radio to channel Will talking to his mother.
  Fact 0.52: Joyce demolishes the wall with an axe but reveals only her front porch.
  Fact 0.53: Nancy discovers a figure behind Barb in Jonathan's photo.
  Fact 0.54: Jonathan realizes the figure matches his mother's description of the creature: a humanoid figure with long arms and no face.
  Fact 0.55: Nancy tells the police about her missing friend.
  Fact 0.56: Nancy subsequently admits to her mother that she slept with Steve.
  Fact 0.57: When Nancy tells Steve about the police, she is irritated that he cares only about not getting in trouble with his father.
  Fact 0.58: Hopper confronts the state trooper who found Will's body and beats him until he admits he was ordered to lie.
  Fact 0.59: Hopper goes to the morgue and finds that Will's supposed body is a dummy.
  Fact 0.60: Hopper breaks into Hawkins laboratory.
  Fact 0.61: Chapter Five: The Flea and the Acrobat was directed by The Duffer Brothers and written by Alison Tatlock.
  Fact 0.62: Chapter Five: The Flea and the Acrobat was released on July 15, 2016.
  Fact 0.63: Will's father Lonnie assures Joyce her experiences are hallucinations.
  Fact 0.64: Hopper discovers a portal in the basement of the lab but is knocked out by the lab's guards.
  Fact 0.65: Hopper wakes at home and discovers a hidden microphone.
  Fact 0.66: Sheriff's deputies inform Hopper that Barb's car was found.
  Fact 0.67: Hopper finds the state's involvement suspicious.
  Fact 0.68: The boys determine Will is trapped in an alternate dimension that Eleven calls the "Upside Down."
  Fact 0.69: The boys determine Will is being pursued by a monster similar to a Demogorgon.
  Fact 0.70: After Will's funeral, science teacher Mr. Clarke tells the boys that a high-energy spacetime tear could create a passage between dimensions.
  Fact 0.71: Using their compasses, the boys search for a disruption in the Earth's electromagnetic field.
  Fact 0.72: Eleven remembers encountering the creature during a lab experiment.
  Fact 0.73: Afraid, Eleven redirects the compasses.
  Fact 0.74: Lucas notices and confronts Eleven.
  Fact 0.75: Mike defends Eleven, and he and Lucas fight.
  Fact 0.76: Eleven telekinetically flings Lucas off Mike.
  Fact 0.77: While Dustin and Mike tend to the unconscious Lucas, Eleven disappears.
  Fact 0.78: In the woods, Nancy and Jonathan find a wounded deer.
  Fact 0.79: The creature drags the deer away, and Nancy follows the trail of blood to a portal to the Upside Down.
  Fact 0.80: Nancy crawls through the portal and discovers the creature feasting upon the deer.
  Fact 0.81: Nancy snaps a twig and draws the creature's attention.
  Fact 0.82: Chapter Six: The Monster was directed by The Duffer Brothers and written by Jessie Nickson-Lopez.
  Fact 0.83: Chapter Six: The Monster was released on July 15, 2016.
  Fact 0.84: Jonathan pulls Nancy back through the portal and saves her from the monster.
  Fact 0.85: Later, Steve sees Nancy and Jonathan together through her bedroom window and assumes they are dating.
  Fact 0.86: Nancy and Jonathan resolve to kill the monster and purchase supplies from an army surplus store.
  Fact 0.87: During a fist fight with Steve, Jonathan inadvertently assaults a police officer and is arrested.
  Fact 0.88: Joyce and Hopper track down Terry Ives, who is catatonic and tended by her sister Becky.
  Fact 0.89: Becky explains that Terry underwent Project MKUltra testing while unknowingly pregnant.
  Fact 0.90: Becky explains that Terry believed her daughter Jane was kidnapped by Brenner at birth.
  Fact 0.91: Joyce and Hopper conclude Eleven is likely Jane.
  Fact 0.92: Eleven shoplifts Eggo waffles.
  Fact 0.93: Eleven recalls being asked to contact the monster and accidentally opening the portal to the Upside Down in the laboratory basement.
  Fact 0.94: While searching for Eleven, Mike and Dustin are ambushed by school bullies Troy and James.
  Fact 0.95: Troy holds Dustin at knifepoint and demands Mike make a suicidal jump into the lake where Will's body was discovered.
  Fact 0.96: Mike jumps but is levitated to safety by Eleven, who scares away the bullies.
  Fact 0.97: Lucas realizes agents from the laboratory are heading to Mike's house.
  Fact 0.98: Chapter Seven: The Bathtub was directed by The Duffer Brothers and written by Justin Doble.
  Fact 0.99: Chapter Seven: The Bathtub was released on July 15, 2016.
  Fact 0.100: Lucas warns Mike via walkie-talkie that government agents are en route.
  Fact 0.101: Mike, Dustin, and Eleven flee.
  Fact 0.102: Eleven telekinetically flips a van that blocks their path, and the kids escape.
  Fact 0.103: Lucas reconciles with Mike and Eleven.
  Fact 0.104: Joyce and Hopper are called to the police station because of Jonathan's assault.
  Fact 0.105: Jonathan and Nancy reveal their knowledge of the monster to Joyce and Hopper.
  Fact 0.106: The group contacts Mike and his friends, and they rendezvous.
  Fact 0.107: The group asks Eleven to search for Will and Barb, but her earlier feats have left her too weak to find them.
  Fact 0.108: To amplify Eleven's powers, the group breaks into the middle school and builds a sensory-deprivation tank.
  Fact 0.109: Eleven finds Barb's corpse with a slug-like creature crawling out of her mouth.
  Fact 0.110: With Joyce's help, Eleven finds Will alive, hiding in the Upside Down "Castle Byers," his backyard fort.
  Fact 0.111: Hopper and Joyce break into Hawkins Laboratory but are apprehended by security guards.
  Fact 0.112: Nancy and Jonathan resolve to kill the monster and steal their hunting gear back from the police station.
  Fact 0.113: In the Upside Down, the monster breaks into the fort where Will is hiding.
  Fact 0.114: Chapter Eight: The Upside Down was directed by The Duffer Brothers, with story by Paul Dichter and teleplay by The Duffer Brothers.
  Fact 0.115: Chapter Eight: The Upside Down was released on July 15, 2016.
  Fact 0.116: Hopper gives up Eleven's location so that he and Joyce can enter the Upside Down.
  Fact 0.117: Hopper and Joyce find Will unconscious in the Upside-Down town library.
  Fact 0.118: Hopper and Joyce pull a tendril out of Will's throat, perform CPR, and revive him.
  Fact 0.119: Nancy and Jonathan booby-trap the Byers' home, then cut their hands to attract the monster with their blood.
  Fact 0.120: Steve arrives, intending to apologize to Jonathan about their fight, but the monster attacks.
  Fact 0.121: The monster springs the trap but escapes back to the Upside Down.
  Fact 0.122: Eleven and the boys hide in the middle school.
  Fact 0.123: Mike asks Eleven to a school dance, the Snow Ball, and kisses her.
  Fact 0.124: The military storms the school to re-capture Eleven, but she kills them and collapses.
  Fact 0.125: As Brenner cradles Eleven, the monster attacks, apparently killing him.
  Fact 0.126: The boys hide Eleven in a classroom.
  Fact 0.127: The monster corners the boys and Eleven, but Eleven pins it against a wall.
  Fact 0.128: Eleven says goodbye to Mike, disintegrates the monster, and vanishes.
  Fact 0.129: Will is hospitalized and reunited with his mother, brother, and friends.
  Fact 0.130: Hopper is reluctantly picked up by a black car.
  Fact 0.131: One month later, Nancy and Steve are back together, and both are friends with Jonathan.
  Fact 0.132: Hopper leaves Eggo waffles and other food in a box in the woods.
  Fact 0.133: Will coughs up a slug-like creature and has a vision of the world as the Upside Down, but he hides this from his family.

Retrieved items:
<item 1>
Stranger Things
1 Introduced in season one 2.3. 2 Introduced in season two 3 Episodes 3.1 Season 1 ( 2016 ) 3.2 Season 2 ( 2017 ) 3.3 Season 3 4 Production 4.1 Development 4.2
Writing 4.3 Casting 4.4 Filming 4.5 Visual effects 4.6 Music 4.6. 1 Soundtracks 5 Release 5.1 Home media 6 Reception 6.1 Audience viewership 6.2 Critical
reception 6.2. 1 Season 1 6.2. 2 Season 2 6.3 Commentary 6.4 Awards and nominations 7 Other media 7.1 Beyond Stranger Things 7.2 Mobile video game 7.3
PlayStation VR game 7.4 Others 8 References 9 External links Premise Stranger Things is set in the fictional rural town of Hawkins , Indiana , during the early
1980s . The nearby Hawkins National Laboratory ostensibly performs scientific research for the United States Department of Energy , but secretly does
experiments into the paranormal and supernatural , including those that involve human test subjects .
</item 1>

<item 2>
Stranger Things
Episodes Season 1 ( 2016 ) No . overall No. in season Title Directed by Written by Original release date `` Chapter One : The Vanishing of Will Byers '' The
Duffer Brothers The Duffer Brothers July 15 , 2016 ( 2016 - 07 - 15 ) On November 6 , 1983 , in a U.S. Department of Energy laboratory in the town of Hawkins ,
Indiana , a scientist is attacked by an unseen creature . While bicycling home from a Dungeons & Dragons session with his friends , 12 - year - old Will Byers
encounters the creature and vanishes . The next day , a buzz - cut young girl wearing a hospital gown steals food from a local diner . The owner , Benny , takes
pity on her and feeds her before calling social services . From a tattoo on her arm , he learns that her name is Eleven . A woman posing as a social worker
arrives and murders Benny . Armed men search the diner for Eleven , but she escapes . Will 's mother Joyce believes she hears Will 's voice on a distorted phone
call , but her phone short circuits .
</item 2>

<item 3>
Graph query row:
episode: Chapter One: The Vanishing of Will Byers
entity: Benny
death_date: None
connection: [{'properties': {}, 'type': 'FROM_CHUNK'}, {'properties': {}, 'type': 'FROM_CHUNK'}]
</item 3>

<item 4>
Stranger Things
Matthew Modine as Martin Brenner , the scientist in charge of Hawkins Laboratory . He is manipulative and remote . He and his team are searching for Eleven . He
reappears in season 2 briefly . Noah Schnapp as Will Byers , the younger brother of Jonathan Byers and the son of Joyce Byers . He is captured by a monster from
the `` Upside Down '' , an alternate dimension discovered by Hawkins Laboratory scientists . Schnapp was promoted to series regular for the second season ,
after recurring in the first . Sadie Sink as Maxine `` Max '' Mayfield / `` Madmax '' , Billy 's younger stepsister , and a tomboy who catches the attention of
both Lucas and Dustin . Joe Keery as Steve Harrington , a popular high school student and the boyfriend of Nancy Wheeler . He ostracizes Jonathan Byers , but
later comes to befriend him . He begins as Nancy 's boyfriend but breaks up with her in season two . Keery was promoted to series regular for the second season
, after recurring in the first . Dacre Montgomery as Billy Hargrove , Max 's violent and unpredictable older stepbrother .
</item 4>

<item 5>
Stranger Things
Gaten Matarazzo as Dustin Henderson , one of Will 's friends . His cleidocranial dysplasia causes him to lisp . In the second season , he is proud of his new
front teeth and is attracted to Max . Caleb McLaughlin as Lucas Sinclair , one of Will 's friends . He is wary of Eleven but later befriends her . In season two
, he is one of Max 's love interests . Natalia Dyer as Nancy Wheeler , daughter of Karen and Ted and older sister of Mike . In season one , she is Steve
Harrington 's girlfriend . In season two , she becomes Jonathan Byers ' girlfriend . Charlie Heaton as Jonathan Byers , the older brother of Will Byers and the
son of Joyce Byers . He is a quiet teenager , an outsider at school , and an aspiring photographer . He is close with his mother and brother , and he becomes
the boyfriend of Nancy Wheeler . Cara Buono as Karen Wheeler , mother of teenaged Nancy , middle schooler Mike , and toddler Holly . Matthew Modine as Martin
Brenner , the scientist in charge of Hawkins Laboratory .
</item 5>

<item 6>
Stranger Things
Will 's brother Jonathan investigates the woods where Will went missing . Hearing screaming , he runs to help but finds only Steve , Nancy , and their friends
roughhousing around Steve 's swimming pool . He secretly photographs them . Barb , left alone by the poolside , vanishes . Joyce receives another call from Will
, hears music from his room , and sees something coming through the wall . `` Chapter Three : Holly , Jolly '' Shawn Levy Jessica Mecklenburg July 15 , 2016 (
2016 - 07 - 15 ) Barb wakes up in an empty and decaying pool . She screams for Nancy and attempts to climb out but is dragged down by something unseen . Joyce
strings Christmas lights around her home to talk to Will , who can turn them on and off . Hawkins Lab director Dr. Martin Brenner allows Hopper to view the lab
's security camera footage . Hopper realizes it is fake . He and his deputies research Hawkins Lab , Brenner , and a woman named Terry Ives who claimed her
daughter was taken by Brenner .
</item 6>

<item 7>
Stranger Things
As Brenner cradles her , the monster attacks , apparently killing him . The boys hide Eleven in a classroom . The monster corners them , but Eleven pins it
against a wall . She says goodbye to Mike , disintegrates the monster , and vanishes . Will is hospitalized and reunited with his mother , brother , and friends
. Hopper is reluctantly picked up by a black car . One month later , Nancy and Steve are back together , and both are friends with Jonathan . Hopper leaves Eggo
waffles and other food in a box in the woods . Will coughs up a slug - like creature and has a vision of the world as the Upside Down , but he hides this from
his family . Season 2 ( 2017 ) No . overall No. in season Title Directed by Written by Original release date 9 `` Chapter One : MADMAX '' The Duffer Brothers
The Duffer Brothers October 27 , 2017 ( 2017 - 10 - 27 ) On October 28 , 1984 , Kali and her gang use her psychic powers to escape the police . Her wrist tattoo
marks her as `` 008 . ''
</item 7>

<item 8>
Stranger Things
Inadvertently , they have created a portal to an alternate dimension called `` the Upside Down '' . The influence of the Upside Down starts to affect the
unknowing residents of Hawkins in calamitous ways . The first season begins in November 1983 , when Will Byers is abducted by a creature from the Upside Down .
His mother , Joyce , and the town 's police chief , Jim Hopper , search for Will . At the same time , a young psychokinetic girl called Eleven escapes from the
laboratory and assists Will 's friends , Mike , Dustin , and Lucas , in their own efforts to find Will . The second season is set a year later , starting in
October 1984 . Will has been rescued , but few know of the details of the events . When it is discovered that Will is still being influenced by entities from
the Upside Down , his friends and family learn there is a larger threat to their universe from the Upside Down .
</item 8>

<item 9>
Stranger Things
Cast and characters The Duffer Brothers , Shawn Levy and the main cast of season 2 of Stranger Things at the 2017 San Diego Comic - Con International Main
Winona Ryder as Joyce Byers , the mother of Will and Jonathan Byers . She is divorced from Lonnie Byers . David Harbour as Jim Hopper , chief of Hawkins Police
Department . After his young daughter died of cancer , Hopper got divorced and lapsed into alcoholism . Finn Wolfhard as Mike Wheeler , son of Karen and Ted ,
younger brother of Nancy , and one of three friends of Will Byers . He is an intelligent and conscientious student and is committed to his friends . He develops
romantic feelings for Eleven . Millie Bobby Brown as Eleven ( `` El '' ) , a young girl with psychokinetic abilities and a limited vocabulary . Her real name is
Jane , the biological daughter of Terry Ives . After escaping from Hawkins Laboratory , where experiments were being performed on her , she befriends Mike ,
Dustin , and Lucas . She develops romantic feelings for Mike . Gaten Matarazzo as Dustin Henderson , one of Will 's friends .
</item 9>

<item 10>
Graph query row:
episode: Chapter One: The Vanishing of Will Byers
entity: scientist
death_date: None
connection: [{'properties': {}, 'type': 'FROM_CHUNK'}, {'properties': {}, 'type': 'FROM_CHUNK'}]
</item 10>
=== END ITEM id=53992e94be9a9b89 ===

=== ITEM id=9aeadfcbc087932f ===
Gold sentences and facts:
Sentence 0: Within eukaryotic cells , DNA is organized into long structures called chromosomes . During cell division these chromosomes are duplicated in the process of DNA replication , providing each cell its own complete set of chromosomes . Eukaryotic organisms ( animals , plants , fungi and protists ) store most of their DNA inside the cell nucleus and some of their DNA in organelles , such as mitochondria or chloroplasts . In contrast prokaryotes ( bacteria and archaea ) store their DNA only in the cytoplasm . Within the eukaryotic chromosomes , chromatin proteins such as histones compact and organize DNA . These compact structures guide the interactions between DNA and other proteins , helping control which parts of the DNA are transcribed .
  Fact 0.0: Within eukaryotic cells, DNA is organized into long structures called chromosomes.
  Fact 0.1: During cell division, chromosomes are duplicated in the process of DNA replication.
  Fact 0.2: DNA replication during cell division provides each cell its own complete set of chromosomes.
  Fact 0.3: Eukaryotic organisms (animals, plants, fungi and protists) store most of their DNA inside the cell nucleus.
  Fact 0.4: Eukaryotic organisms store some of their DNA in organelles, such as mitochondria or chloroplasts.
  Fact 0.5: Prokaryotes (bacteria and archaea) store their DNA only in the cytoplasm, in contrast to eukaryotes.
  Fact 0.6: Within eukaryotic chromosomes, chromatin proteins such as histones compact and organize DNA.
  Fact 0.7: The compact structures formed by chromatin proteins guide the interactions between DNA and other proteins.
  Fact 0.8: The compact structures formed by chromatin proteins help control which parts of the DNA are transcribed.
Sentence 1: Genomic DNA is tightly and orderly packed in the process called DNA condensation , to fit the small available volumes of the cell . In eukaryotes , DNA is located in the cell nucleus , with small amounts in mitochondria and chloroplasts . In prokaryotes , the DNA is held within an irregularly shaped body in the cytoplasm called the nucleoid . The genetic information in a genome is held within genes , and the complete set of this information in an organism is called its genotype . A gene is a unit of heredity and is a region of DNA that influences a particular characteristic in an organism . Genes contain an open reading frame that can be transcribed , and regulatory sequences such as promoters and enhancers , which control transcription of the open reading frame .
  Fact 1.0: Genomic DNA is tightly and orderly packed in a process called DNA condensation.
  Fact 1.1: DNA condensation fits genomic DNA into the small available volumes of the cell.
  Fact 1.2: In eukaryotes, DNA is located in the cell nucleus, with small amounts in mitochondria and chloroplasts.
  Fact 1.3: In prokaryotes, DNA is held within an irregularly shaped body in the cytoplasm called the nucleoid.
  Fact 1.4: The genetic information in a genome is held within genes.
  Fact 1.5: The complete set of genetic information in an organism is called its genotype.
  Fact 1.6: A gene is a unit of heredity.
  Fact 1.7: A gene is a region of DNA that influences a particular characteristic in an organism.
  Fact 1.8: Genes contain an open reading frame that can be transcribed.
  Fact 1.9: Genes contain regulatory sequences such as promoters and enhancers.
  Fact 1.10: The regulatory sequences such as promoters and enhancers control transcription of the open reading frame.
Sentence 2: DNA usually occurs as linear chromosomes in eukaryotes , and circular chromosomes in prokaryotes . The set of chromosomes in a cell makes up its genome ; the human genome has approximately 3 billion base pairs of DNA arranged into 46 chromosomes . The information carried by DNA is held in the sequence of pieces of DNA called genes . Transmission of genetic information in genes is achieved via complementary base pairing . For example , in transcription , when a cell uses the information in a gene , the DNA sequence is copied into a complementary RNA sequence through the attraction between the DNA and the correct RNA nucleotides . Usually , this RNA copy is then used to make a matching protein sequence in a process called translation , which depends on the same interaction between RNA nucleotides . In alternative fashion , a cell may simply copy its genetic information in a process called DNA replication . The details of these functions are covered in other articles ; here the focus is on the interactions between DNA and other molecules that mediate the function of the genome .
  Fact 2.0: DNA usually occurs as linear chromosomes in eukaryotes.
  Fact 2.1: DNA usually occurs as circular chromosomes in prokaryotes.
  Fact 2.2: The set of chromosomes in a cell makes up its genome.
  Fact 2.3: The human genome has approximately 3 billion base pairs of DNA.
  Fact 2.4: The human genome's DNA is arranged into 46 chromosomes.
  Fact 2.5: The information carried by DNA is held in the sequence of pieces of DNA called genes.
  Fact 2.6: Transmission of genetic information in genes is achieved via complementary base pairing.
  Fact 2.7: In transcription, when a cell uses the information in a gene, the DNA sequence is copied into a complementary RNA sequence.
  Fact 2.8: The copying in transcription happens through attraction between the DNA and the correct RNA nucleotides.
  Fact 2.9: The RNA copy is usually then used to make a matching protein sequence in a process called translation.
  Fact 2.10: Translation depends on the same interaction between RNA nucleotides.
  Fact 2.11: In an alternative fashion, a cell may simply copy its genetic information in a process called DNA replication.
  Fact 2.12: The details of these functions are covered in other articles.
  Fact 2.13: The focus of this text is on the interactions between DNA and other molecules that mediate the function of the genome.

Retrieved items:
<item 1>
DNA
`` Jump up ^ Miescher F ( 1871 ) . `` Ueber die chemische Zusammensetzung der Eiterzellen '' ( On the chemical composition of pus cells ) . Medicinisch -
chemische Untersuchungen ( in German ) . 4 : 441 -- 460 . Ich habe mich daher später mit meinen Versuchen an die ganzen Kerne gehalten , die Trennung der Körper
, die ich einstweilen ohne weiteres Präjudiz als lösliches und unlösliches Nuclein bezeichnen will , einem günstigeren Material überlassend . ( Therefore , in
my experiments I subsequently limited myself to the whole nucleus , leaving to a more favorable material the separation of the substances , that for the present
, without further prejudice , I will designate as soluble and insoluble nuclear material ( `` Nuclein '' ) Jump up ^ Dahm R ( January 2008 ) . `` Discovering
DNA : Friedrich Miescher and the early years of nucleic acid research '' . Human Genetics . 122 ( 6 ) : 565 -- 81 .
</item 1>

<item 2>
DNA
However , identification can be complicated if the scene is contaminated with DNA from several people . DNA profiling was developed in 1984 by British
geneticist Sir Alec Jeffreys , and first used in forensic science to convict Colin Pitchfork in the 1988 Enderby murders case . The development of forensic
science and the ability to now obtain genetic matching on minute samples of blood , skin , saliva , or hair has led to re-examining many cases . Evidence can
now be uncovered that was scientifically impossible at the time of the original examination . Combined with the removal of the double jeopardy law in some
places , this can allow cases to be reopened where prior trials have failed to produce sufficient evidence to convince a jury . People charged with serious
crimes may be required to provide a sample of DNA for matching purposes . The most obvious defense to DNA matches obtained forensically is to claim that
cross-contamination of evidence has occurred . This has resulted in meticulous strict handling procedures with new cases of serious crime . DNA profiling is
also used successfully to positively identify victims of mass casualty incidents , bodies or body parts in serious accidents , and individual victims in mass
war graves , via matching to family members .
</item 2>

<item 3>
DNA
In this way , the base on the old strand dictates which base appears on the new strand , and the cell ends up with a perfect copy of its DNA . Extracellular
nucleic acids Naked extracellular DNA ( eDNA ) , most of it released by cell death , is nearly ubiquitous in the environment . Its concentration in soil may be
as high as 2 μg / L , and its concentration in natural aquatic environments may be as high at 88 μg / L. Various possible functions have been proposed for eDNA
: it may be involved in horizontal gene transfer ; it may provide nutrients ; and it may act as a buffer to recruit or titrate ions or antibiotics .
Extracellular DNA acts as a functional extracellular matrix component in the biofilms of several bacterial species . It may act as a recognition factor to
regulate the attachment and dispersal of specific cell types in the biofilm ; it may contribute to biofilm formation ; and it may contribute to the biofilm 's
physical strength and resistance to biological stress . Cell - free fetal DNA is found in the blood of the mother , and can be sequenced to determine a great
deal of information about the developing fetus .
</item 3>

<item 4>
DNA
In contrast prokaryotes ( bacteria and archaea ) store their DNA only in the cytoplasm . Within the eukaryotic chromosomes , chromatin proteins such as histones
compact and organize DNA . These compact structures guide the interactions between DNA and other proteins , helping control which parts of the DNA are
transcribed . DNA was first isolated by Friedrich Miescher in 1869 . Its molecular structure was first identified by James Watson and Francis Crick at the
Cavendish Laboratory within the University of Cambridge in 1953 , whose model - building efforts were guided by X-ray diffraction data acquired by Raymond
Gosling , who was a post-graduate student of Rosalind Franklin . DNA is used by researchers as a molecular tool to explore physical laws and theories , such as
the ergodic theorem and the theory of elasticity . The unique material properties of DNA have made it an attractive molecule for material scientists and
engineers interested in micro - and nano - fabrication . Among notable advances in this field are DNA origami and DNA - based hybrid materials .
</item 4>

<item 5>
DNA
DNA profiling is also used in DNA paternity testing to determine if someone is the biological parent or grandparent of a child with the probability of parentage
is typically 99.99 % when the alleged parent is biologically related to the child . Normal DNA sequencing methods happen after birth , but there are new methods
to test paternity while a mother is still pregnant . DNA enzymes or catalytic DNA Further information : Deoxyribozyme Deoxyribozymes , also called DNAzymes or
catalytic DNA , are first discovered in 1994 . They are mostly single stranded DNA sequences isolated from a large pool of random DNA sequences through a
combinatorial approach called in vitro selection or systematic evolution of ligands by exponential enrichment ( SELEX ) . DNAzymes catalyze variety of chemical
reactions including RNA - DNA cleavage , RNA - DNA ligation , amino acids phosphorylation - dephosphorylation , carbon - carbon bond formation , and etc .
DNAzymes can enhance catalytic rate of chemical reactions up to 100,000,000,000-fold over the uncatalyzed reaction .
</item 5>

<item 6>
DNA
The DNA sequence may be aligned with other DNA sequences to identify homologous sequences and locate the specific mutations that make them distinct . These
techniques , especially multiple sequence alignment , are used in studying phylogenetic relationships and protein function . Data sets representing entire
genomes ' worth of DNA sequences , such as those produced by the Human Genome Project , are difficult to use without the annotations that identify the locations
of genes and regulatory elements on each chromosome . Regions of DNA sequence that have the characteristic patterns associated with protein - or RNA - coding
genes can be identified by gene finding algorithms , which allow researchers to predict the presence of particular gene products and their possible functions in
an organism even before they have been isolated experimentally . Entire genomes may also be compared , which can shed light on the evolutionary history of
particular organism and permit the examination of complex evolutionary events . DNA nanotechnology The DNA structure at left ( schematic shown ) will self -
assemble into the structure visualized by atomic force microscopy at right . DNA nanotechnology is the field that seeks to design nanoscale structures using the
molecular recognition properties of DNA molecules . Image from Strong , 2004 .
</item 6>

<item 7>
DNA
This can be used in studies ranging from ecological genetics to anthropology ; For example , DNA evidence is being used to try to identify the Ten Lost Tribes
of Israel . Information storage Main article : DNA digital data storage In a paper published in Nature in January 2013 , scientists from the European
Bioinformatics Institute and Agilent Technologies proposed a mechanism to use DNA 's ability to code information as a means of digital data storage . The group
was able to encode 739 kilobytes of data into DNA code , synthesize the actual DNA , then sequence the DNA and decode the information back to its original form
, with a reported 100 % accuracy . The encoded information consisted of text files and audio files . A prior experiment was published in August 2012 . It was
conducted by researchers at Harvard University , where the text of a 54,000 - word book was encoded in DNA . Moreover , in living cells , the storage can be
turned active by enzymes . Light - gated protein domains fused to DNA processing enzymes are suitable for that task in vitro . Fluorescent exonucleases can
transmit the output according to the nucleotide they have read .
</item 7>

<item 8>
DNA
PMID 9068179 . Jump up ^ Jeffreys AJ , Wilson V , Thein SL ( 1985 ) . `` Individual - specific ' fingerprints ' of human DNA '' . Nature . 316 ( 6023 ) : 76 --
9 . Bibcode : 1985Natur. 316 ... 76J . doi : 10.1038 / 316076a0 . PMID 2989708 . Jump up ^ Colin Pitchfork -- first murder conviction on DNA evidence also
clears the prime suspect Forensic Science Service Accessed 23 December 2006 Jump up ^ `` DNA Identification in Mass Fatality Incidents '' . National Institute
of Justice . September 2006 . Archived from the original on 12 November 2006 . Jump up ^ `` Paternity Blood Tests That Work Early in a Pregnancy '' New York
Times June 20 , 2012 Archived 24 June 2017 at the Wayback Machine . ^ Jump up to : Breaker RR , Joyce GF ( December 1994 ) . `` A DNA enzyme that cleaves RNA ''
. Chemistry & Biology .
</item 8>

<item 9>
DNA
Biological functions Location of eukaryote nuclear DNA within the chromosomes . DNA usually occurs as linear chromosomes in eukaryotes , and circular
chromosomes in prokaryotes . The set of chromosomes in a cell makes up its genome ; the human genome has approximately 3 billion base pairs of DNA arranged into
46 chromosomes . The information carried by DNA is held in the sequence of pieces of DNA called genes . Transmission of genetic information in genes is achieved
via complementary base pairing . For example , in transcription , when a cell uses the information in a gene , the DNA sequence is copied into a complementary
RNA sequence through the attraction between the DNA and the correct RNA nucleotides . Usually , this RNA copy is then used to make a matching protein sequence
in a process called translation , which depends on the same interaction between RNA nucleotides . In alternative fashion , a cell may simply copy its genetic
information in a process called DNA replication . The details of these functions are covered in other articles ; here the focus is on the interactions between
DNA and other molecules that mediate the function of the genome .
</item 9>

<item 10>
DNA
Modern biology and biochemistry make intensive use of these techniques in recombinant DNA technology . Recombinant DNA is a man - made DNA sequence that has
been assembled from other DNA sequences . They can be transformed into organisms in the form of plasmids or in the appropriate format , by using a viral vector
. The genetically modified organisms produced can be used to produce products such as recombinant proteins , used in medical research , or be grown in
agriculture . DNA profiling Further information : DNA profiling Forensic scientists can use DNA in blood , semen , skin , saliva or hair found at a crime scene
to identify a matching DNA of an individual , such as a perpetrator . This process is formally termed DNA profiling , but may also be called `` genetic
fingerprinting '' . In DNA profiling , the lengths of variable sections of repetitive DNA , such as short tandem repeats and minisatellites , are compared
between people . This method is usually an extremely reliable technique for identifying a matching DNA . However , identification can be complicated if the
scene is contaminated with DNA from several people .
</item 10>
=== END ITEM id=9aeadfcbc087932f ===

=== ITEM id=25ce6f61d3f09f1a ===
Gold sentences and facts:
Sentence 0: German autobahns have no federally mandated speed limit for some classes of vehicles . However , limits are posted ( and enforced ) in areas that are urbanized , substandard , accident - prone , or under construction . On speed - unrestricted stretches , an advisory speed limit ( Richtgeschwindigkeit ) of 130 kilometres per hour ( 81 mph ) applies . While going faster is not illegal as such in the absence of a speed limit , it can cause an increased liability in the case of an accident ; courts have ruled that an `` ideal driver '' who is exempt from absolute liability for `` inevitable '' tort under the law would not exceed Richtgeschwindigkeit .
  Fact 0.0: German autobahns have no federally mandated speed limit for some classes of vehicles.
  Fact 0.1: Speed limits are posted in areas that are urbanized.
  Fact 0.2: Speed limits are posted in areas that are substandard.
  Fact 0.3: Speed limits are posted in areas that are accident-prone.
  Fact 0.4: Speed limits are posted in areas that are under construction.
  Fact 0.5: Posted speed limits are enforced.
  Fact 0.6: On speed-unrestricted stretches, an advisory speed limit applies.
  Fact 0.7: The advisory speed limit is called Richtgeschwindigkeit.
  Fact 0.8: The advisory speed limit is 130 kilometres per hour (81 mph).
  Fact 0.9: Going faster is not illegal as such in the absence of a speed limit.
  Fact 0.10: Going faster can cause an increased liability in the case of an accident.
  Fact 0.11: Courts have ruled that an 'ideal driver' is exempt from absolute liability for 'inevitable' tort under the law.
  Fact 0.12: Courts have ruled that an 'ideal driver' would not exceed Richtgeschwindigkeit.

Retrieved items:
<item 1>
Autobahn
Truck stop Scandinavian Park off the A 7 Rest area Dammer Berge on the A 1 Rest areas and truck stops are marked several times , starting several kilometres in
advance , and with larger signs that often include icons announcing what kinds of facilities travellers can expect . Speed limits ( edit ) Autobahn with three
separate lanes in each direction and an emergency lane On the autobahns there is an advisory speed limit ( Richtgeschwindigkeit ) of 130 km / h ( unless
otherwise regulated by signs ) . The `` limits no longer apply '' ( Ende aller Streckenverbote ) sign , indicating a return to the default speed , while lifting
all other limits as well ( all limits are indicated by round signs with red border ) . GPS tracks colored according to speed show considerable speed differences
at an autobahn crossing Germany 's autobahns are famous for being among the few public roads in the world without blanket speed limits for cars and motorcycles
. As such , they are important German cultural identifiers , `` ... often mentioned in hushed , reverential tones by motoring enthusiasts and looked at with a
mix of awe and terror by outsiders . '' Some speed limits are implemented on different autobahns .
</item 1>

<item 2>
Autobahn
She spun across two lanes and smashed into a bank of trees . Jump up ^ `` StVO -- Einzelnorm '' . Gesetze-im-internet.de . Retrieved 2014 - 03 - 21 . Further
reading ( edit ) Vahrenkamp , Richard ( 2010 ) . The German Autobahn 1920 -- 1945 : Hafraba Visions and Mega Projects . Josef Eul Verlag GmbH . Zeller , Thomas
( 2010 ) . Driving Germany : The Landscape of the German Autobahn , 1930 -- 1970 . Berghahn Books . External links ( edit ) Wikimedia Commons has media related
to Bundesautobahn . German website with descriptions of all autobahn routes and exits ( in English ) English - language website that discusses all aspects of
the autobahn Geographic data related to Autobahn at OpenStreetMap Autobahn system in Germany Major routes A 1 A 2 A 3 A 4 A 5 A 6 A 7 A 8 A 9 Regional routes A
10 A 11 A 12 A 13 A 14 A 15 A
</item 2>

<item 3>
Autobahn
These limiters can be deactivated , so speeds up to 300 km / h ( 190 mph ) might arise on the German autobahn , but due to other traffic , such speeds are
generally not attainable except during certain times like between 10 p.m. and 6 a.m. or on Sundays ( when trucks drivers have to rest by law ) . Furthermore ,
there are certain autobahn sections which are known for having light traffic , making such speeds attainable during most days ( especially some of those located
in Eastern Germany ) . Most unlimited sections of the autobahn are located outside densely populated areas . Vehicles with a top speed less than 60 km / h ( 37
mph ) ( such as quads , low - end microcars , and agricultural / construction equipment ) are not allowed to use the autobahn , nor are motorcycles and scooters
with low engine capacity regardless of top speed ( mainly applicable to mopeds which are typically limited to 25 or 45 km / h anyway ) . To comply with this
limit , heavy - duty trucks in Germany ( e.g. mobile cranes , tank transporters etc . )
</item 3>

<item 4>
Autobahn
The official German term is Bundesautobahn ( plural Bundesautobahnen , abbreviated BAB ) , which translates as `` federal motorway '' . The literal meaning of
the word Bundesautobahn is `` Federal Auto ( mobile ) Track '' . German autobahns have no federally mandated speed limit for some classes of vehicles . However
, limits are posted ( and enforced ) in areas that are urbanized , substandard , accident - prone , or under construction . On speed - unrestricted stretches ,
an advisory speed limit ( Richtgeschwindigkeit ) of 130 kilometres per hour ( 81 mph ) applies . While going faster is not illegal as such in the absence of a
speed limit , it can cause an increased liability in the case of an accident ; courts have ruled that an `` ideal driver '' who is exempt from absolute
liability for `` inevitable '' tort under the law would not exceed Richtgeschwindigkeit . A 2008 estimate reported that 52 % of the autobahn network had only
the advisory speed limit , 15 % had temporary speed limits due to weather or traffic conditions , and 33 % had permanent speed limits .
</item 4>

<item 5>
Autobahn
Der Spiegel . 2008 - 04 - 09 . Retrieved 2014 - 03 - 21 . From now on , the maximum speed allowed is 120 kilometers per hour ( 75 miles per hour ) . In
practical terms , the change is not a big one -- it only affects 11 kilometers ( 6.8 miles ) of autobahn connecting the cities of Bremen and Bremerhaven . The
rest of the tiny city - state 's 49 kilometers ( 30.5 miles ) of autobahn has long had speed restrictions aimed at fighting congestion , noise and pollution .
Jump up ^ Manfred Kühnappel . `` Als erstes Bundesland : Bremen führt Tempolimit ein '' . RP ONLINE . Retrieved 14 April 2016 . Jump up ^ Chris Wevers . ``
Tensions Rise in The Heartland of German Automotive Industry '' . GTspirit . Retrieved 14 April 2016 . Jump up ^ `` bw-fairs.de : The automotive industry in
Baden - Württemberg '' . Retrieved 14 April 2016 .
</item 5>

<item 6>
Autobahn
While this initiative is primarily a part of the SPD 's general strategic outline for the future and , according to practices , not necessarily meant to affect
immediate government policy , the proposal had stirred up a debate once again ; Germany 's chancellor since 2005 , Angela Merkel , and leading cabinet members
expressed outspoken disapproval of such a measure . In 2008 , the Social Democratic - Green Party coalition controlling Germany 's smallest State , the paired
City - State of Bremen & Bremerhaven , imposed a 120 - kilometre - per - hour ( 75 mph ) limit on its last 11 kilometres ( 6.8 mi ) of speed - unlimited
autobahn in hopes of leading other States to do likewise . In 2011 , the first ever Green minister - president of any German state , Winfried Kretschmann of
Baden - Württemberg initially argued for a similar , state - level 120 kilometres per hour ( 75 mph ) limit .
</item 6>

<item 7>
Autobahn
A 24 , connecting Berlin and Hamburg ) . The system is as follows : A 10 to A 19 are in eastern Germany ( Berlin , Saxony - Anhalt , parts of Saxony and
Brandenburg ) A 20 to A 29 are in northern and northeastern Germany A 30 to A 39 are in Lower Saxony ( northwestern Germany ) and Thuringia A 40 to A 49 are in
the Rhine - Ruhr to Frankfurt Rhine - Main A 50 to A 59 are in the Lower Rhine region to Cologne A 60 to A 69 are in Rhineland - Palatinate , Saarland , Hesse
and northern Baden - Württemberg A 70 to A 79 are in Thuringia , northern Bavaria and parts of Saxony A 80 to A 89 are in Baden - Württemberg A 90 to A 99 are
in ( southern ) Bavaria There are also some very short autobahns built just for local traffic ( e.g. ring roads or the A 555 from Cologne to Bonn ) that usually
have three digits for numbering .
</item 7>

<item 8>
Autobahn
However , Baden - Württemberg is an important location for the German motor industry , including the headquarters of Daimler AG and Porsche ; the ruling
coalition ultimately decided against a state - level limit on its 675 kilometres ( 419 mi ) of speed - unlimited roads -- arguing for nationwide speed limit
instead . In 2014 , the conservative - liberal ruling coalition of Saxony confirmed its rejection of a general speed limit on autobahns , instead advocating
dynamic traffic controls where appropriate . Between 2010 and 2014 in the State of Hesse , transportation ministers Dieter Posch and his successor Florian
Rentsch , both members of the Free Democratic Party , removed or raised speed limits on several sections of autobahn following regular 5 - year reviews of speed
limit effectiveness ; some sections just prior to the installation of Tarek Al - Wazir ( Green Party ) as Transportation Minister in January 2014 as part of an
uneasy CDU - green coalition government .
</item 8>

<item 9>
Autobahn
Certain stretches have lower speed limits during wet weather . Some areas have a speed limit of 120 km / h ( 75 mph ) in order to reduce noise pollution during
overnight hours ( usually 10pm -- 6am ) or because of increased traffic during daytime ( 6am -- 8pm ) . Dynamic traffic signs on an Autobahn Some limits were
imposed to reduce pollution and noise . Limits can also be temporarily put into place through dynamic traffic guidance systems that display the according
message . More than half of the total length of the German autobahn network has no speed limit , about one third has a permanent limit , and the remaining parts
have a temporary or conditional limit . Some cars with very powerful engines can reach speeds of well over 300 km / h ( 190 mph ) . Major German car
manufacturers , except Porsche , follow a gentlemen 's agreement by electronically limiting the top speeds of their cars -- with the exception of some top of
the range models or engines -- to 250 km / h ( 155 mph ) .
</item 9>

<item 10>
Autobahn
Vorerst , denn grundsätzlich unterstützt Verkehrsminister Winfried Hermann ( Grüne ) Initiativen auf Bundesebene , die eine flächendeckende
Geschwindigkeitsbegrenzung auf Autobahnen in Deutschland vorsehen , wie aus seiner Antwort auf eine Landtagsanfrage der FDP - Fraktion hervorgeht ... Auf den
2078 Kilometer langen Autobahnen in Baden - Württemberg bestehen nach Angaben des Ministers auf 726 Kilometern Tempolimits. 65 Prozent seien frei ; auf 35
Prozent sind Geschwindigkeiten bis 120 Stundenkilometer erlaubt . ( The Red - Green ruling coalition plans no general speed limit on autobahns , instead
supporting the proposal of Transportation Minister Winfried Hermann ( Green Party ) to impose a Federal limit , said the government in response to a
parliamentary question by the Free Democratic Party ... Of the 2,078 carriageway - kilometres of Baden - Württemberg autobahns , 726 kilometres have speed
limits .
</item 10>
=== END ITEM id=25ce6f61d3f09f1a ===
