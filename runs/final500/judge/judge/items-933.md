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

=== ITEM id=5a78bd03a42f4ad5 ===
Gold sentences and facts:
Sentence 0: Angular cheilitis can be caused by infection , irritation , or allergies . Infections include by fungi such as Candida albicans and bacteria such as Staph . aureus . Irritants include poorly fitting dentures , licking the lips or drooling , mouth breathing resulting in a dry mouth , sun exposure , overclosure of the mouth , smoking , and minor trauma . Allergies may include substances like toothpaste , makeup , and food . Often a number of factors are involved . Other factors may include poor nutrition or poor immune function . Diagnosis may be helped by testing for infections and patch testing for allergies .
  Fact 0.0: Angular cheilitis can be caused by infection.
  Fact 0.1: Angular cheilitis can be caused by irritation.
  Fact 0.2: Angular cheilitis can be caused by allergies.
  Fact 0.3: Infections that cause angular cheilitis include infection by fungi such as Candida albicans.
  Fact 0.4: Infections that cause angular cheilitis include infection by bacteria such as Staph. aureus.
  Fact 0.5: Irritants that cause angular cheilitis include poorly fitting dentures.
  Fact 0.6: Irritants that cause angular cheilitis include licking the lips or drooling.
  Fact 0.7: Irritants that cause angular cheilitis include mouth breathing resulting in a dry mouth.
  Fact 0.8: Irritants that cause angular cheilitis include sun exposure.
  Fact 0.9: Irritants that cause angular cheilitis include overclosure of the mouth.
  Fact 0.10: Irritants that cause angular cheilitis include smoking.
  Fact 0.11: Irritants that cause angular cheilitis include minor trauma.
  Fact 0.12: Allergies that cause angular cheilitis may include substances like toothpaste.
  Fact 0.13: Allergies that cause angular cheilitis may include substances like makeup.
  Fact 0.14: Allergies that cause angular cheilitis may include substances like food.
  Fact 0.15: Often a number of factors are involved in angular cheilitis.
  Fact 0.16: Other factors may include poor nutrition.
  Fact 0.17: Other factors may include poor immune function.
  Fact 0.18: Diagnosis of angular cheilitis may be helped by testing for infections.
  Fact 0.19: Diagnosis of angular cheilitis may be helped by patch testing for allergies.

Retrieved items:
<item 1>
Angular cheilitis
Angular cheilitis - wikipedia Angular cheilitis Jump to : navigation , search Angular cheilitis Synonyms Rhagades , perlèche , cheilosis , angular cheilosis ,
commissural cheilitis , angular stomatitis Bilateral angular cheilitis in an elderly individual with false teeth , iron deficiency anemia and dry mouth
Pronunciation / kaɪˈlaɪtɪs / Specialty Dermatology Symptoms Redness , skin breakdown and crusting at the corner of the mouth Usual onset Children , 30s to 60s
Duration Days to years Causes Infection , irritation , allergies Treatment Based on cause , barrier cream Frequency 0.7 % of the population Angular cheilitis (
AC ) , is inflammation of one or both corners of the mouth . Often the corners are red with skin breakdown and crusting . It can also be itchy or painful . The
condition can last for days to years . Angular cheilitis is a type of cheilitis ( inflammation of the lips ) . Angular cheilitis can be caused by infection ,
irritation , or allergies . Infections include by fungi such as Candida albicans and bacteria such as Staph .
</item 1>

<item 2>
Angular cheilitis
Infections include by fungi such as Candida albicans and bacteria such as Staph . aureus . Irritants include poorly fitting dentures , licking the lips or
drooling , mouth breathing resulting in a dry mouth , sun exposure , overclosure of the mouth , smoking , and minor trauma . Allergies may include substances
like toothpaste , makeup , and food . Often a number of factors are involved . Other factors may include poor nutrition or poor immune function . Diagnosis may
be helped by testing for infections and patch testing for allergies . Treatment for angular cheilitis is typically based on the underlying causes along with the
use of a barrier cream . Frequently an antifungal and antibacterial cream is also tried . Angular cheilitis is a fairly common problem , with estimates that it
affects 0.7 % of the population . It occurs most often in the 30s to 60s , although is also relatively common in children . In the developing world , iron and
vitamin deficiencies are a common cause .
</item 2>

<item 3>
Angular cheilitis
The corner of the mouth is normally exposed to saliva more than any other part of the lips . Reduced lower facial height ( vertical dimension or facial support
) is usually caused by edentulism ( tooth loss ) , or wearing worn down , old dentures or ones which are not designed optimally . This results in overclosure of
the mandible ( collapse of the jaws ) , which extenuates the angular skin folds at the corners of the mouth , in effect creating an intertriginous skin crease .
The tendency of saliva to pool in these areas is increased , constantly wetting the area , which may cause tissue maceration and favors the development of a
yeast infection . As such , angular cheilitis is more commonly seen in edentulous people ( people without any teeth ) . It is by contrast uncommon in persons
who retain their natural teeth . Angular cheilitis is also commonly seen in denture wearers . Angular cheilitis is present in about 30 % of people with denture
- related stomatitis .
</item 3>

<item 4>
Angular cheilitis
Such examples include people with anorexia nervosa . Other disorders may cause lip enlargement ( e.g. orofacial granulomatosis ) , which alters the local
anatomy and extenuates the skin folds at the corners of the mouth . More still may be involved because they affect the immune system , allowing normally
harmless organisms like Candida to become pathogenic and cause an infection . Xerostomia ( dry mouth ) is thought to account for about 5 % of cases of AC .
Xerostomia itself has many possible causes , but commonly the cause may be side effects of medications , or conditions such as Sjögren 's syndrome . Conversely
, conditions which cause drooling or sialorrhoea ( excessive salivation ) can cause angular cheilitis by creating a constant wet environment in the corners of
the mouth . About 25 % of people with Down syndrome appear to have AC . This is due to relative macroglossia , an apparently large tongue in a small mouth ,
which may constantly stick out of the mouth causing maceration of the corners of the mouth with saliva .
</item 4>

<item 5>
Angular cheilitis
In the developing world , iron and vitamin deficiencies are a common cause . Contents ( hide ) 1 Signs and symptoms 2 Causes 2.1 Infection 2.2 Irritation
contact dermatitis 2.3 Nutritional deficiencies 2.4 Systemic disorders 2.5 Drugs 2.6 Allergic contact dermatitis 3 Diagnosis 3.1 Classification 4 Management 5
Prognosis 6 Epidemiology 7 References 8 External links Signs and symptoms ( edit ) Angular cheilitis -- a fissure running in the corner of the mouth with
reddened , irritated facial skin adjacent . A fairly mild case of angular cheilitis extending onto the facial skin in a young person ( affected area is within
the black oval ) . Angular cheilitis is a fairly non specific term which describes the presence of an inflammatory lesion in a particular anatomic site ( i.e.
the corner of the mouth ) . As there are different possible causes and contributing factors from one person to the next , the appearance of the lesion is
somewhat variable . The lesions are more commonly symmetrically present on both sides of the mouth , but sometimes only one side may be affected .
</item 5>

<item 6>
Angular cheilitis
In chronic angular cheilitis , there may be suppuration ( pus formation ) , exfoliation ( scaling ) and formation of granulation tissue . Sometimes contributing
factors can be readily seen , such as loss of lower face height from poorly made or worn dentures , which results in mandibular overclosure ( `` collapse of
jaws '' ) . If there is a nutritional deficiency underlying the condition , various other signs and symptoms such as glossitis ( swollen tongue ) may be present
. In people with angular cheilitis who wear dentures , often there may be erythematous mucosa underneath the denture ( normally the upper denture ) , an
appearance consistent with denture - related stomatitis . Typically the lesions give symptoms of soreness , pain , pruritus ( itching ) or burning or a raw
feeling . Causes ( edit ) Angular cheilitis is thought to be multifactorial disorder of infectious origin , with many local and systemic predisposing factors .
The sores in angular cheilitis are often infected with fungi ( yeasts ) , bacteria , or a combination thereof ; this may represent a secondary , opportunistic
infection by these pathogens .
</item 6>

<item 7>
Angular cheilitis
A cold sore at the corner of the mouth behaves similarly to elsewhere on the lips , and follows a pattern of vesicle ( blister ) formation followed by rupture
leaving a crusted sore which resolves in about 7 -- 10 days , and recurs in the same spot periodically , especially during periods of stress . Rather than
utilizing antifungal creams , angular herpes simplex is treated in the same way as a cold sore , with topical antiviral drugs such as aciclovir . Irritation
contact dermatitis ( edit ) A famous sketch by Leonardo da Vinci in preparation to depict the face of Judas Iscariot in The Last supper . The subject shows
overclosure of the jaws and loss of facial support around the mouth . Pronounced skin folds extending from the corner of the mouth . Main article : Irritant
contact dermatitis 22 % of cases of angular cheilitis are due to irritants . Saliva contains digestive enzymes , which may have a degree of digestive action on
tissues if they are left in contact . The corner of the mouth is normally exposed to saliva more than any other part of the lips .
</item 7>

<item 8>
Angular cheilitis
The term `` cheilocandidiasis '' describes exfoliative ( flaking ) lesions of the lips and the skin around the lips , and is caused by a superficial candidal
infection due to chronic lip licking . Less severe cases occur during cold , dry weather , and is a form of chapped lips . Individuals may lick their lips in an
attempt to provide a temporary moment of relief , only serving to worsen the condition . The sunscreen in some types of lip balm degrades over time into an
irritant . Using expired lipbalm can initiate mild angular cheilitis , and when the person applies more lipbalm to alleviate the cracking , it only aggravates
it . Because of the delayed onset of contact dermatitis and the recovery period lasting days to weeks , people typically do not make the connection between the
causative agent and the symptoms . Nutritional deficiencies ( edit ) Several different nutritional deficiency states of vitamins or minerals have been linked to
AC . It is thought that in about 25 % of people with AC , iron deficiency or deficiency of B vitamins are involved . Nutritional deficiencies may be a more
common cause of AC in Third World countries .
</item 8>

<item 9>
Angular cheilitis
It is thought that reduced vertical dimension of the lower face may be a contributing factor in up to 11 % of elderly persons with angular cheilitis and in up
to 18 % of denture wearers who have angular cheilitis . Reduced vertical dimension can also be caused by tooth migration , wearing orthodontic appliances , and
elastic tissue damage caused by ultraviolet light exposure and smoking . Habits or conditions that keep the corners of the mouth moist might include chronic lip
licking , thumb sucking ( or sucking on other objects such as pens , pipes , lollipops ) , dental cleaning ( e.g. flossing ) , chewing gum , hypersalivation ,
drooling and mouth breathing . Some consider habitual lip licking or picking to be a form of nervous tic , and do not consider this to be true angular cheilitis
, instead calling it perlèche ( derived from the French word pourlècher meaning `` to lick one 's lips '' ) , or `` factitious cheilitis '' is applied to this
habit .
</item 9>

<item 10>
Angular cheilitis
Examples include cocaine , methamphetamines , heroin , and hallucinogens . Allergic contact dermatitis ( edit ) See also : Allergic contact cheilitis , Allergic
contact dermatitis , and Allergic contact stomatitis Patch test Allergic reactions may account for about 25 -- 34 % of cases of generalized cheilitis ( i.e. ,
inflammation not confined to the angles of the mouth ) . It is unknown how frequently allergic reactions are responsible for cases of angular cheilitis , but
any substance capable of causing generalized allergic cheilitis may present involving the corners of the mouth alone . Examples of potential allergens include
substances that may be present in some types of lipstick , toothpaste , acne products , cosmetics , chewing gum , mouthwash , foods , dental appliances , and
materials from dentures or mercury containing amalgam fillings . It is usually impossible to tell the difference between irritant contact dermatitis and
allergic contact dermatitis without a patch test . Diagnosis ( edit ) Photographic Comparison of : 1 ) a Canker Sore - inside the mouth , 2 ) Herpes , 3 )
Angular Cheilitis and 4 ) Chapped Lips .
</item 10>
=== END ITEM id=5a78bd03a42f4ad5 ===

=== ITEM id=6ed0ed3197a5a97b ===
Gold sentences and facts:
Sentence 0: Windows Defender , officially called Windows Defender Antivirus in Windows 10 Creators Update , is an anti-malware component of Microsoft Windows.It was first released as a free antispyware program download for Windows XP , shipped with Windows Vista and Windows 7 and made into a full antivirus program replacing Microsoft Security Essentials as part of Windows 8 and later versions .
  Fact 0.0: Windows Defender is an anti-malware component of Microsoft Windows.
  Fact 0.1: Windows Defender is officially called Windows Defender Antivirus in Windows 10 Creators Update.
  Fact 0.2: Windows Defender was first released as a free antispyware program download for Windows XP.
  Fact 0.3: Windows Defender shipped with Windows Vista.
  Fact 0.4: Windows Defender shipped with Windows 7.
  Fact 0.5: Windows Defender was made into a full antivirus program as part of Windows 8 and later versions.
  Fact 0.6: Windows Defender replaced Microsoft Security Essentials as part of Windows 8 and later versions.

Retrieved items:
<item 1>
Windows Defender
General availability ( edit ) On October 24 , 2006 , Microsoft released Windows Defender . It supports Windows XP and Windows Server 2003 ; however , unlike the
betas , it does not run on Windows 2000 . Conversion to antivirus ( edit ) Windows Defender was released with Windows Vista and Windows 7 , serving as their
built - in antispyware component . In Windows Vista and Windows 7 , Windows Defender was superseded by Microsoft Security Essentials , an antivirus product from
Microsoft which provided protection against a wider range of malware . Upon installation , Microsoft Security Essentials disabled and replaced Windows Defender
. In Windows 8 , Microsoft upgraded Windows Defender into an antivirus program very similar to Microsoft Security Essentials for Windows 7 and using the same
virus definition updates . MSE itself does not run on Windows versions beyond 7 . In Windows 8 and Windows 10 , Windows Defender is on by default . It switches
itself off upon installation of a third party anti-virus package . Starting with Windows 10 , Microsoft began to transfer the control of Windows Defender out of
its native client . Initially , its `` Settings '' dialog box was replaced by dedicated page in the Settings app .
</item 1>

<item 2>
Windows Defender
The same update introduced a new flat logo . History ( edit ) Beta ( edit ) Microsoft AntiSpyware Beta 1 ( Version 1.0. 701 ) Windows Defender is based on GIANT
AntiSpyware , which was originally developed by GIANT Company Software , Inc . The company 's acquisition was announced by Microsoft on December 16 , 2004 .
While the original GIANT AntiSpyware supported older Windows versions , support for the Windows 9x line of operating systems was later dropped . The first beta
version of Microsoft AntiSpyware was released on January 6 , 2005 and was basically a repackaged GIANT AntiSpyware . More builds were released in 2005 , with
the last Beta 1 refresh released on November 21 , 2005 . At the 2005 RSA Security conference , Chief Software Architect and co-founder of Microsoft , Bill Gates
, announced that Windows Defender ( which was known as Microsoft AntiSpyware prior to November 4 , 2005 ) would be made available free of charge to all validly
licensed Windows 2000 , Windows XP , and Windows Server 2003 users to help secure their systems against the increasing malware threat .
</item 2>

<item 3>
Windows Defender
Initially , its `` Settings '' dialog box was replaced by dedicated page in the Settings app . In Windows 10 Creators Update , Windows Defender is renamed
Windows Defender Antivirus to distinguish it from Windows Defender Security Center . The latter has become the default avenue to interface with Windows Defender
. While there is no shortcut on the Start menu for Windows Defender 's native client , it can still run . Advanced features ( edit ) Windows Defender
successfully blocks the EICAR test file Real - time protection In the Windows Defender options , the user can configure real - time protection options . Browser
integration Integration with Internet Explorer and Microsoft Edge enables files to be scanned as they are downloaded to detect malicious software inadvertently
downloaded . Although it does not integrate with non-Microsoft web browsers , Windows Defender scans for malicious downloaded files as part of its real - time
protection . Windows 10 's Anniversary Update introduced Limited Periodic Scanning , which optionally allows Windows Defender to scan a system periodically if
another antivirus app is installed . It also introduced Block at First Sight , which uses machine learning to predict whether a file is malicious . Windows
Vista - specific functionality ( edit ) Windows Defender had additional functionality in Windows Vista which was removed in subsequent versions of Windows .
</item 3>

<item 4>
Windows Defender
The first and last categories carry a link to learn more about the particular item , and the second category invites users to submit the program to Microsoft
SpyNet for analysis by community members . The Software Explorer feature has been removed from Windows Defender in Windows 7 . Notification of startup programs
that run as an administrator Windows Defender in Windows Vista automatically blocks all startup items that require administrator privileges to run ( this is
considered suspicious behavior for a startup item ) . This automatic blocking is related to the User Account Control functionality in Windows Vista , and
requires users to manually run each of these startup items each time they log in if they desire the item to run at startup . User interface In Windows Vista ,
it is possible to close the window and have the program run in the system tray while a scan is running . However , in Windows 7 , this functionality was removed
and the window must remain open while a scan is running . Windows Defender offline ( edit ) Windows Defender Offline ( formerly known as Standalone System
Sweeper Beta ) is a bootable standalone antimalware program that runs from a bootable disk and is designed to scan infected systems while their operating
systems are offline .
</item 4>

<item 5>
Windows Defender
wikipedia.org/w/index.php?title=Windows_Defender&oldid=818651625 '' Categories : Microsoft software Microsoft Windows security technology Windows - only
freeware Spyware removal Windows components 2006 software Windows security software Hidden categories : CS1 German - language sources ( de ) CS1 Japanese -
language sources ( ja ) Articles needing additional references from July 2009 All articles needing additional references Talk Search Contents About Wikipedia
Čeština Dansk Deutsch Ελληνικά Español فارسی Français 한국어 Bahasa Indonesia Italiano עברית Bahasa Melayu Nederlands 日本 語 Norsk Polski Português Română Русский
සිංහල Simple English Suomi Svenska Tagalog தமிழ் ไทย Türkçe Українська Tiếng Việt 中文 Edit links This page was last edited on 4 January 2018 , at 20 : 09 . Text
is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply .
</item 5>

<item 6>
Windows Defender
Windows Defender - wikipedia Windows Defender Jump to : navigation , search Not to be confused with Windows Defender Security Center . This article needs
additional citations for verification . Please help improve this article by adding citations to reliable sources . Unsourced material may be challenged and
removed . ( July 2009 ) ( Learn how and when to remove this template message ) Windows Defender A component of Microsoft Windows Windows Defender Antivirus , as
appears in the Windows 10 Creators Update Details Type Antivirus software on Windows 8 and later ; formerly spyware removal software Included with Windows Vista
and later , Windows Server 2016 Also available for Windows XP and Windows Server 2003 Support status Same as the version of Windows with which it is integrated
Related components Microsoft Security Essentials Windows Defender Security Center Windows Defender , officially called Windows Defender Antivirus in Windows 10
Creators Update , is an anti-malware component of Microsoft Windows.It was first released as a free antispyware program download for Windows XP , shipped with
Windows Vista and Windows 7 and made into a full antivirus program replacing Microsoft Security Essentials as part of Windows 8 and later versions .
</item 6>

<item 7>
Windows Defender
Contents ( hide ) 1 Basic features 2 History 2.1 Beta 2.2 General availability 2.3 Conversion to antivirus 3 Advanced features 4 Windows Vista - specific
functionality 5 Windows Defender Offline 6 Mitigated security vulnerability 7 See also 8 References 9 External links Basic features ( edit ) Before Windows 8 ,
Windows Defender protected against spyware . It included a number of real - time security agents that monitored several common areas of Windows for changes
which might have been caused by spyware . It also included the ability to easily remove installed ActiveX software. Windows Defender featured integrated support
for Microsoft SpyNet that allows users to report to Microsoft what they consider to be spyware , and what applications and device drivers they allow to be
installed on their system . Protection against viruses was added in Windows 8 ; Windows Defender in Windows 8 resembles Microsoft Security Essentials ( MSE )
and uses the same virus definitions . In Windows 10 , Windows Defender settings are controlled by the Settings app , and the Settings button opens the Settings
app . In the Windows 10 Anniversary Update , toast notifications appear to announce the results of a scan , even if no viruses are found .
</item 7>

<item 8>
Windows Defender
Windows Defender ( Beta 2 ) was released on February 13 , 2006 . It featured the program 's new name and a significant user interface redesign . The core engine
was rewritten in C++ , unlike the original GIANT - developed one , which was written in Visual Basic . This improved the application 's performance . Also ,
since beta 2 , the program works as a Windows service , unlike earlier releases , which enables the application to protect the computer even when a user is not
logged on . Beta 2 also requires Windows Genuine Advantage validation . However , Windows Defender ( Beta 2 ) did not contain some of the tools found in
Microsoft AntiSpyware ( Beta 1 ) ; Microsoft removed the System Inoculation , Secure Shredder and System Explorer tools found in MSAS ( Beta 1 ) as well as the
Tracks Eraser tool , which allowed users to easily delete many different types of temporary files related to Internet Explorer 6 , including HTTP cookies , web
cache , and Windows Media Player playback history . Microsoft later released German and Japanese versions of Windows Defender ( Beta 2 ) . General availability
( edit ) On October 24 , 2006 , Microsoft released Windows Defender .
</item 8>

<item 9>
Windows Defender
`` Windows Defender Beta 2 Review : Paul Thurrott 's SuperSite for Windows '' . SuperSite for Windows . Retrieved 26 November 2013 . Jump up ^ `` Windows
Defender : Startseite '' ( in German ) . Microsoft Corporation . Archived from the original on 30 January 2009 . Retrieved 8 May 2011 . Jump up ^ `` マイクロソフト
セキュリティ At Home '' ( in Japanese ) . Microsoft Corporation . Archived from the original on 18 January 2010 . Retrieved 8 May 2011 . Jump up ^ Thurrott , Paul (
24 October 2006 ) . `` Finally , Microsoft Ships Windows Defender '' . Windows IT Pro . Retrieved 8 May 2011 . Jump up ^ Thurrott , Paul ( 18 June 2009 ) . ``
Microsoft Security Essentials Public Beta '' . Paul Thurrott 's SuperSite for Windows . Retrieved 8 May 2011 . Jump up ^ Hau , Kevin ( 23 June 2009 ) .
</item 9>

<item 10>
Windows Defender
Since Windows 10 Anniversary Update , offline functionality is integrated into the regular Windows Defender program . Mitigated Security vulnerability ( edit )
On 5 May 2017 , Tavis Ormandy , a vulnerability researcher for Google , discovered a security vulnerability in the JavaScript analysis module ( NScript ) of
Microsoft Antimalware Engine ( MsMpEngine ) that impacted Windows Defender , Microsoft Security Essentials and System Center Endpoint Protection . By 8 May 2017
, Microsoft had released a patch to all affected systems . Ars Technica commended Microsoft for its unprecedented patching speed and said that the disaster had
been averted . See also ( edit ) Security and safety features new to Windows Vista References ( edit ) Jump up ^ `` Windows Defender '' . Download Center .
Microsoft . 23 May 2007 . Archived from the original on 29 April 2012 . ^ Jump up to : `` What is Windows Defender ? '' . Computer Hope . Retrieved 31 December
2017 . Jump up ^ Shultz , Greg . `` Windows Defender : Past , present , and future '' . Retrieved 13 June 2017 .
</item 10>
=== END ITEM id=6ed0ed3197a5a97b ===

=== ITEM id=51435e2cfc72a21e ===
Gold sentences and facts:
Sentence 0: No . Name Field Affiliation Date of Appointment Date of Retirement Roopa Ganguly Art Bharatiya Janata Party 04 - Oct - 2016 03 - Oct - 2022 Sambhaji Raje Social work Bharatiya Janata Party 07 - Jun - 2016 03 - May - 2022 Suresh Gopi Art Bharatiya Janata Party 25 - Apr - 2016 24 - Apr - 2022 Subramanian Swamy Economics Bharatiya Janata Party 25 - Apr - 2016 24 - Apr - 2022 5 Narendra Jadhav Economics Nominated 25 - Apr - 2016 24 - Apr - 2022 6 Mary Kom Sport Nominated 25 - Apr - 2016 24 - Apr - 2022 7 Swapan Dasgupta Journalism Nominated 25 - Apr - 2016 24 - Apr - 2022 8 K.T.S. Tulsi Law Nominated 25 - Feb - 2014 24 - Feb - 2020 9 K. Parasaran Law Nominated 09 - Jun - 2012 28 - Jun - 2018 10 Rekha Art Nominated 27 - Apr - 2012 26 - Apr - 2018 11 Sachin Tendulkar Social service Nominated 27 - Apr - 2012 26 - Apr - 2018 12 Anu Aga Business Nominated 27 - Apr - 2012 26 - Apr - 2018
  Fact 0.0: Roopa Ganguly's field is Art and affiliation is Bharatiya Janata Party, with appointment on 04-Oct-2016 and retirement on 03-Oct-2022.
  Fact 0.1: Sambhaji Raje's field is Social work and affiliation is Bharatiya Janata Party, with appointment on 07-Jun-2016 and retirement on 03-May-2022.
  Fact 0.2: Suresh Gopi's field is Art and affiliation is Bharatiya Janata Party, with appointment on 25-Apr-2016 and retirement on 24-Apr-2022.
  Fact 0.3: Subramanian Swamy's field is Economics and affiliation is Bharatiya Janata Party, with appointment on 25-Apr-2016 and retirement on 24-Apr-2022.
  Fact 0.4: Narendra Jadhav (No. 5) has field Economics and affiliation Nominated, with appointment on 25-Apr-2016 and retirement on 24-Apr-2022.
  Fact 0.5: Mary Kom (No. 6) has field Sport and affiliation Nominated, with appointment on 25-Apr-2016 and retirement on 24-Apr-2022.
  Fact 0.6: Swapan Dasgupta (No. 7) has field Journalism and affiliation Nominated, with appointment on 25-Apr-2016 and retirement on 24-Apr-2022.
  Fact 0.7: K.T.S. Tulsi (No. 8) has field Law and affiliation Nominated, with appointment on 25-Feb-2014 and retirement on 24-Feb-2020.
  Fact 0.8: K. Parasaran (No. 9) has field Law and affiliation Nominated, with appointment on 09-Jun-2012 and retirement on 28-Jun-2018.
  Fact 0.9: Rekha (No. 10) has field Art and affiliation Nominated, with appointment on 27-Apr-2012 and retirement on 26-Apr-2018.
  Fact 0.10: Sachin Tendulkar (No. 11) has field Social service and affiliation Nominated, with appointment on 27-Apr-2012 and retirement on 26-Apr-2018.
  Fact 0.11: Anu Aga (No. 12) has field Business and affiliation Nominated, with appointment on 27-Apr-2012 and retirement on 26-Apr-2018.

Retrieved items:
<item 1>
List of nominated members of Rajya Sabha
Kasturirangan 2003 -- 2009 102 -- Narayan Singh Manaklao 2003 -- 2009 103 -- Vidya Niwas Misra 2003 -- 2009 104 -- Chandan Mitra 2003 -- 2009 105 Dara Singh
2003 -- 2009 106 Kapila Vatsyayan 2006 -- 2012 107 Ram Jethmalani 2006 -- 2009 108 -- Shobhana Bhartia 2006 -- 2012 109 Shyam Benegal 2006 -- 2012 110 M.S.
Swaminathan 2007 -- 2013 111 -- C. Rangarajan 2008 -- 2010 112 -- H.K.
</item 1>

<item 2>
List of nominated members of Rajya Sabha
Narayana Reddy 1997 -- 2003 93 Mrinal Sen 1997 -- 2003 94 -- Chaudhary Harmohan Singh Yadav 1997 -- 2003 95 -- Nanaji Deshmukh 1999 -- 2005 96 Lata Mangeshkar
1999 -- 2005 97 -- Fali Nariman 1999 -- 2005 98 -- Cho Ramaswamy 1999 -- 2005 99 Hema Malini 2003 -- 2009 100 -- Bimal Jalan 2003 -- 2009 101 K.
</item 2>

<item 3>
List of nominated members of Rajya Sabha
Setalvad 1966 -- 1972 35 G. Sankara Kurup 1968 -- 1972 36 Joachim Alva 1968 -- 1974 37 S. Nurul Hasan 1968 -- 1974 38 -- K. Ramiah 1968 -- 1974 39 -- Ganga
Sharan Sinha 1968 -- 1974 40 -- Maragatham Chandrasekhar 1970 -- 1988 41 -- Umashankar Joshi 1970 -- 1976 42 -- Rasheeduddin Khan 1970 -- 1982 43 -- Vidya
Prakash Dutt 1971 -- 1980 44 -- Abu Abraham 1972 -- 1978 45 -- Pramathanath Bishi 1972 -- 1978 46 -- C.K.
</item 3>

<item 4>
List of nominated members of Rajya Sabha
Tulsi 2014 - 2020 125 Subramanian Swamy 2016 - 2022 126 MC Mary Kom 2016 - 2022 127 Narendra Jadhav 2016 - 2022 129 Suresh Gopi 2016 - 2022 130 -- Swapan
Dasgupta 2016 - 2022 131 -- Sambhaji Raje 2016 - 2022 132 -- Roopa Ganguly 2016 - 2022 References ( edit ) Nominated Members since 1952 Narender Kumar Current
list of Nominated Members Specific Jump up ^ `` Statewise List '' . 164.100. 47.5 . Retrieved 12 June 2016 . Jump up ^
http://164.100.47.5/Newmembers/partymemberlist.aspx ^ Jump up to : `` Statewise Retirement '' . 164.100. 47.5 . Retrieved 12 June 2016 .
</item 4>

<item 5>
List of nominated members of Rajya Sabha
Daphtary 1972 -- 1978 47 -- Habib Tanvir 1972 -- 1978 48 -- Krishna Kriplani 1974 -- 1980 49 Lokesh Chandra 1974 -- 1986 50 -- Scato Swu 1974 -- 1986 51 -- B.N.
Banerjee 1976 -- 1982 52 -- Malcolm Adiseshiah 1978 -- 1984 53 -- Fathema Ismail 1978 -- 1984 54 -- Pandurang Dharmaji Jadhav 1978 -- 1984 55 -- Bhagwati Charan
Varma 1978 -- 1984 56 Nargis Dutt 1980 -- 1986 57 Khushwant Singh 1980 -- 1986 58 -- Asima Chatterjee 1982 -- 1990 59 Sivaji Ganesan 1982 -- 1986 60 --
Hayatullah Ansari 1982 -- 1988 61 -- Madan Bhatia 1982 -- 1994 62 -- V.N.
</item 5>

<item 6>
List of nominated members of Rajya Sabha
Krishnamachari 1961 -- 1967 24 -- R.R. Diwakar 1962 -- 1968 25 -- Gopal Singh 1962 -- 1968 26 -- M. Ajmal Khan 1964 -- 1972 27 -- Satyavrata Siddhantalankar
1964 -- 1968 28 Shakuntala Paranjpye 1964 -- 1970 29 -- Badri Nath Prasad 1964 -- 1970 30 -- G. Ramachandran 1964 -- 1970 31 -- M.N. Kaul 1964 -- 1972 32 --
Harivanshrai Bachchan 1966 -- 1972 33 -- Dhananjay Ramchandra Gadgil 1966 -- 1972 34 -- M.C.
</item 6>

<item 7>
Graph query row:
person: Rukmini Devi Arundale
organization: Rajya Sabha
relationship: MEMBER_OF
role: None
period: None
term: None
fact: None
context: None
relationship_detail: None
relation_detail: None
</item 7>

<item 8>
List of nominated members of Rajya Sabha
No . Name Field Affiliation Date of Appointment Date of Retirement Roopa Ganguly Art Bharatiya Janata Party 04 - Oct - 2016 03 - Oct - 2022 Sambhaji Raje Social
work Bharatiya Janata Party 07 - Jun - 2016 03 - May - 2022 Suresh Gopi Art Bharatiya Janata Party 25 - Apr - 2016 24 - Apr - 2022 Subramanian Swamy Economics
Bharatiya Janata Party 25 - Apr - 2016 24 - Apr - 2022 5 Narendra Jadhav Economics Nominated 25 - Apr - 2016 24 - Apr - 2022 6 Mary Kom Sport Nominated 25 - Apr
- 2016 24 - Apr - 2022 7 Swapan Dasgupta Journalism Nominated 25 - Apr - 2016 24 - Apr - 2022 8 K.T.S.
</item 8>

<item 9>
List of nominated members of Rajya Sabha
Dutta 1993 -- 1999 85 -- Maulana Habibur Rahman Nomani 1993 -- 1999 86 Nirmala Deshpande 1997 -- 1999 and 2004 -- 2010 87 Shabana Azmi 1997 -- 2003 88 -- P.
Selvie Das 1997 -- 2003 89 -- Kartar Singh Duggal 1997 -- 2003 90 Kuldip Nayyar 1997 -- 2003 91 -- Raja Ramanna 1997 -- 2003 92 C.
</item 9>

<item 10>
List of nominated members of Rajya Sabha
Sr. No . Image Name Tenure -- Alladi Krishnaswamy Iyer 1952 -- 1953 Satyendranath Bose 1952 -- 1960 Prithviraj Kapoor 1952 -- 1960 -- Jagadisan Mohandas
Kumarappa 1952 -- 1954 5 -- Kalidas Nag 1952 -- 1954 6 Rukmini Devi Arundale 1952 -- 1962 7 -- N.R. Malkani 1952 -- 1962 8 -- Sahib Singh Sokhey 1952 -- 1956 9
-- Zakir Hussain 1952 -- 1962 10 -- Maithili Sharan Gupt 1952 -- 1964 11 -- Kakasaheb Kalelkar 1952 -- 1964 12 -- Radha Kumud Mukherjee 1952 -- 1958 13 P.V.
</item 10>
=== END ITEM id=51435e2cfc72a21e ===

=== ITEM id=b4ac7e4944152f73 ===
Gold sentences and facts:
Sentence 0: Group 3 in the periodic table Hydrogen Helium Lithium Beryllium Boron Carbon Nitrogen Oxygen Fluorine Neon Sodium Magnesium Aluminium Silicon Phosphorus Sulfur Chlorine Argon Potassium Calcium Scandium Titanium Vanadium Chromium Manganese Iron Cobalt Nickel Copper Zinc Gallium Germanium Arsenic Selenium Bromine Krypton Rubidium Strontium Yttrium Zirconium Niobium Molybdenum Technetium Ruthenium Rhodium Palladium Silver Cadmium Indium Tin Antimony Tellurium Iodine Xenon Caesium Barium Lanthanum Cerium Praseodymium Neodymium Promethium Samarium Europium Gadolinium Terbium Dysprosium Holmium Erbium Thulium Ytterbium Lutetium Hafnium Tantalum Tungsten Rhenium Osmium Iridium Platinum Gold Mercury ( element ) Thallium Lead Bismuth Polonium Astatine Radon Francium Radium Actinium Thorium Protactinium Uranium Neptunium Plutonium Americium Curium Berkelium Californium Einsteinium Fermium Mendelevium Nobelium Lawrencium Rutherfordium Dubnium Seaborgium Bohrium Hassium Meitnerium Darmstadtium Roentgenium Copernicium Nihonium Flerovium Moscovium Livermorium Tennessine Oganesson alkaline earth metals ← → group 4 IUPAC group number Name by element scandium group CAS group number ( US , pattern A-B-A ) IIIB old IUPAC number ( Europe , pattern A-B ) IIIA ↓ Period Scandium ( Sc ) 21 Transition metal 5 Yttrium ( Y ) 39 Transition metal 6 Lanthanum ( La * ) 57 Lanthanide 7 Actinium ( Ac * ) 89 Actinide * Whether the elements lutetium ( Lu ) and lawrencium ( Lr ) , in period 6 and 7 , are in group 3 is disputed . The grouping used in this article places La and Ac in group 3 , which is the most common form . For other groupings , see group 3 borders . Legend primordial element synthetic element Atomic number color : black = solid
  Fact 0.0: The periodic table has a group 3.
  Fact 0.1: The text lists the element names from Hydrogen through Oganesson.
  Fact 0.2: The text mentions alkaline earth metals.
  Fact 0.3: The article's grouping places lanthanum (La) and actinium (Ac) in group 3.
  Fact 0.4: The article says placing La and Ac in group 3 is the most common form.
  Fact 0.5: Scandium (Sc) has atomic number 21 and is a transition metal.
  Fact 0.6: Yttrium (Y) has atomic number 39, is in period 5, and is a transition metal.
  Fact 0.7: Lanthanum (La) has atomic number 57, is in period 6, and is a lanthanide.
  Fact 0.8: Actinium (Ac) has atomic number 89, is in period 7, and is an actinide.
  Fact 0.9: Whether lutetium (Lu) and lawrencium (Lr), in periods 6 and 7, are in group 3 is disputed.
  Fact 0.10: For other groupings, the article refers readers to group 3 borders.
  Fact 0.11: The scandium group has CAS group number IIIB (US pattern A-B-A).
  Fact 0.12: The scandium group has old IUPAC number IIIA (Europe pattern A-B).
  Fact 0.13: The article has a legend distinguishing primordial elements and synthetic elements.
  Fact 0.14: The article has a legend for atomic number colors, where black means solid.

Retrieved items:
<item 1>
Group 3 element
Three group 3 elements occur naturally , scandium , yttrium , and either lanthanum or lutetium . Lanthanum continues the trend started by two lighter members in
general chemical behavior , while lutetium behaves more similarly to yttrium . While the choice of lutetium would be in accordance with the trend for period 6
transition metals to behave more similarly to their upper periodic table neighbors , the choice of lanthanum is in accordance with the trends in the s - block ,
which the group 3 elements are chemically more similar to . They all are silvery - white metals under standard conditions . The fourth element , either actinium
or lawrencium , has only radioactive isotopes . Actinium , which occurs only in trace amounts , continues the trend in chemical behavior for metals that form
tripositive ions with a noble gas configuration ; synthetic lawrencium is calculated and partially shown to be more similar to lutetium and yttrium . So far ,
no experiments have been conducted to synthesize any element that could be the next group 3 element .
</item 1>

<item 2>
Group 3 element
Jump up ^ Greenwood , N.N. ; Harrington , T.J. ( 1973 ) . The chemistry of the transition elements . Oxford : Clarendon Press . p. 50 . ISBN 0 - 19 - 855435 - 4
. Jump up ^ Aylward , G. ; Findlay , T. ( 2008 ) . SI chemical data ( 6th ed . ) . Milton , Queensland : John Wiley & Sons . ISBN 978 - 0 - 470 - 81638 - 7 .
Jump up ^ Wiberg , N. ( 2001 ) . Inorganic Chemistry . San Diego : Academic Press . p. 119 . ISBN 0 - 12 - 352651 - 5 . Jump up ^ Wulfsberg , G. ( 2006 ) . ``
Periodic table : Trends in the properties of the elements '' . Encyclopedia of Inorganic Chemistry . New York : John Wiley & Sons . p. 3 .
</item 2>

<item 3>
Group 3 element
element ) Thallium Lead Bismuth Polonium Astatine Radon Francium Radium Actinium Thorium Protactinium Uranium Neptunium Plutonium Americium Curium Berkelium
Californium Einsteinium Fermium Mendelevium Nobelium Lawrencium Rutherfordium Dubnium Seaborgium Bohrium Hassium Meitnerium Darmstadtium Roentgenium Copernicium
Nihonium Flerovium Moscovium Livermorium Tennessine Oganesson alkaline earth metals ← → group 4 IUPAC group number Name by element scandium group CAS group
number ( US , pattern A-B-A ) IIIB old IUPAC number ( Europe , pattern A-B ) IIIA ↓ Period Scandium ( Sc ) 21 Transition metal 5 Yttrium ( Y ) 39 Transition
metal 6 Lanthanum ( La * ) 57 Lanthanide 7 Actinium ( Ac * ) 89 Actinide * Whether the elements lutetium ( Lu ) and lawrencium ( Lr ) , in period 6 and 7 , are
in group 3 is disputed .
</item 3>

<item 4>
Group 3 element
. Dordrecht , The Netherlands : Springer Science + Business Media . ISBN 1 - 4020 - 3555 - 1 . Jump up ^ Pyykkö , Pekka ( 2011 ) . `` A suggested periodic table
up to Z ≤ 172 , based on Dirac -- Fock calculations on atoms and ions '' . Physical Chemistry Chemical Physics. 13 ( 1 ) : 161 -- 8 . Bibcode : 2011PCCP ... 13
... 161P . doi : 10.1039 / c0cp01575j . PMID 20967377 . Jump up ^ van der Krogt , Peter . `` Elementymology & Elements Multidict '' . Elements.vanderkrogt.net .
Retrieved 4 July 2011 . Jump up ^ Seaborg , G.T. ( c. 2006 ) . `` transuranium element ( chemical element ) '' . Encyclopædia Britannica . Retrieved 2010 - 03 -
16 .
</item 4>

<item 5>
Group 3 element
^ Jump up to : Barbalace , Kenneth . `` Periodic Table of Elements '' . Environmental Chemistry.com . Retrieved 2007 - 04 - 14 . Jump up ^ `` WebElements
Periodic Table of the Elements '' . Webelements.com . Retrieved 2010 - 04 - 03 . Jump up ^ `` Periodic Table of the Elements '' . International Union of Pure
and Applied Chemistry ( IUPAC ) . Retrieved 2010 - 04 - 03 . ^ Jump up to : `` Visual Elements '' . Royal Society of Chemistry . Retrieved 4 July 2011 . Jump up
^ Dolg , Michael . `` Lanthanides and Actinides '' ( PDF ) . Max - Planck - Institut für Physik komplexer Systeme , Dresden , Germany . CLA01 . Retrieved 4 July
2011 . Jump up ^ IUPAC ( 2015 ) . `` The constitution of group 3 of the periodic table '' . Retrieved 2016 - 11 - 10 . Jump up ^ Greenwood , N.N.
</item 5>

<item 6>
Group 3 element
in period 6 and 7 , are in group 3 is disputed . The grouping used in this article places La and Ac in group 3 , which is the most common form . For other
groupings , see group 3 borders . Legend primordial element synthetic element Atomic number color : black = solid Group 3 is a group of elements in the periodic
table . This group , like other d - block groups , should contain four elements , but it is not agreed what elements belong in the group . Scandium ( Sc ) and
yttrium ( Y ) are always included , but the other two spaces are usually occupied by lanthanum ( La ) and actinium ( Ac ) , or by lutetium ( Lu ) and lawrencium
( Lr ) ; less frequently , it is considered the group should be expanded to 32 elements ( with all the lanthanides and actinides included ) or contracted to
contain only scandium and yttrium . When the group is understood to contain all of the lanthanides , its trivial name is the rare - earth metals .
</item 6>

<item 7>
Group 3 element
If lanthanum and actinium are considered group 3 elements , then element 121 , unbiunium ( Ubu ) , should be the fifth group 3 element . The element is
calculated have electronic configuration of ( Og ) 8s 8p , which is not associated with transition metals , without having a partially filled d - subshell . No
experiments have been performed to create unpenttrium , unbiunium or any element that could be considered the next group 3 element ; however , unbiunium is the
element with the lowest atomic number that has not been tried to be created and thus has chances to be , while unpenttrium , unpentpentium or any other element
considered if preceded by lawrencium is very unlikely to be created due to drip instabilities that imply that the periodic table ends soon after the island of
stability at unbihexium .
</item 7>

<item 8>
Group 3 element
ISBN 0 - 19 - 850341 - 5 . Links to related articles ( edit ) ( hide ) Periodic table ( Large cells ) 5 6 7 8 9 10 11 12 13 14 15 16 17 18 He Li Be O Ne Na Mg
Al Si Cl Ar Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr 5 Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te Xe 6 Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm
Yb Lu Hf Ta Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn 7 Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og Alkali metal
Alkaline earth metal Lanthanide Actinide Transition metal Post - ​ transition metal Metalloid Reactive nonmetal Noble gas Unknown chemical properties Periodic
table Periodic table forms Standard 18 - column 18 - column ,
</item 8>

<item 9>
Group 3 element
Group 3 element - wikipedia Group 3 element Jump to : navigation , search For the other group formerly named Group III ( boron group ) , see Group 13 element .
Group 3 in the periodic table Hydrogen Helium Lithium Beryllium Boron Carbon Nitrogen Oxygen Fluorine Neon Sodium Magnesium Aluminium Silicon Phosphorus Sulfur
Chlorine Argon Potassium Calcium Scandium Titanium Vanadium Chromium Manganese Iron Cobalt Nickel Copper Zinc Gallium Germanium Arsenic Selenium Bromine Krypton
Rubidium Strontium Yttrium Zirconium Niobium Molybdenum Technetium Ruthenium Rhodium Palladium Silver Cadmium Indium Tin Antimony Tellurium Iodine Xenon Caesium
Barium Lanthanum Cerium Praseodymium Neodymium Promethium Samarium Europium Gadolinium Terbium Dysprosium Holmium Erbium Thulium Ytterbium Lutetium Hafnium
Tantalum Tungsten Rhenium Osmium Iridium Platinum Gold Mercury ( element ) Thallium Lead Bismuth Polonium Astatine Radon Francium Radium
</item 9>

<item 10>
Group 3 element
large cells 32 - column , large cells Alternative Alternatives Janet 's left step table Extended Extension beyond the 7th period Fricke model large cells Pyykkö
model Sets of elements By periodic table structure Groups 1 ( Alkali metals ) 2 ( Alkaline earth metals ) 5 6 7 8 9 10 11 12 13 14 15 ( Pnictogens ) 16 (
Chalcogens ) 17 ( Halogens ) 18 ( Noble gases ) Periods 5 6 7 8 + Aufbau Fricke Pyykkö Blocks s - block p - block d - block f - block g - block Aufbau principle
By metallicity Metals Alkali metals Alkaline earth metals Lanthanides Actinides Transition metals Post-transition metals Metalloids Lists of metalloids by
source Dividing line Nonmetals Reactive nonmetals Noble gases Other sets Nutritional elements Platinum - group metals ( PGM ) Rare - earth elements Refractory
metals Precious metals Coinage metals Noble metal Heavy metals Native metals Transuranium elements Transactinide elements Major actinides Minor actinides
Elements Lists By :
</item 10>
=== END ITEM id=b4ac7e4944152f73 ===

=== ITEM id=be2eec9aa07b11cc ===
Gold sentences and facts:
Sentence 0: This article covers the professional life of George W. Bush , the 43rd President of the United States . Prior to his election as President in 2000 , Bush held numerous other positions , including being an oil executive , an owner of the Texas Rangers baseball team , and Governor of Texas . ( Also see Governorship of George W. Bush . )
  Fact 0.0: This article covers the professional life of George W. Bush.
  Fact 0.1: George W. Bush was the 43rd President of the United States.
  Fact 0.2: Bush was elected President in 2000.
  Fact 0.3: Before his election as President, Bush held numerous other positions.
  Fact 0.4: Bush was an oil executive.
  Fact 0.5: Bush was an owner of the Texas Rangers baseball team.
  Fact 0.6: Bush was Governor of Texas.
  Fact 0.7: The article Governorship of George W. Bush is a see-also reference.

Retrieved items:
<item 1>
Professional life of George W. Bush
Bush and who had been a lawyer in James Baker 's firm , Baker Botts . The SEC 's general counsel at the time was James Doty , who had been appointed by
President H.W. Bush and as a lawyer in James Baker 's firm , Baker Botts had represented George W. Bush when arrangements were made to acquire the Texas Rangers
baseball franchise ( although Doty recused himself from the investigation . ) . With Baker Botts representing W. Bush , the Saudi BinLaden family and W. Bush 's
funding conduit James R. Bath , Doty was involved in the frivilous litigation campaign launched in the attempt to intimidate BinLaden middleman James R Bath 's
business partner ( ( Charles W. ( `` Bill '' ) White ) ) into cooperating with the attempted cover - up of secret BinLaden Family funding of W. Bush 's
campaigns and businesses . Bush 's own lawyer was Robert Jordan , who had been `` partners with both Doty and Breeden at Baker Botts and who later became George
W. Bush 's ambassador to Saudi Arabia '' .
</item 1>

<item 2>
Professional life of George W. Bush
Securities and Exchange Commission ( SEC ) ruled that it would not charge Bush with insider trading . On June 22 , Bush sold his 212,140 shares of stock anyway
for a net profit of $848,560 . The very next quarter , Harken announced losses of $23 million , which continued to the end of the year when the stock ``
plummeted from $4 to $1.25 '' . As President , Bush has refused to authorize the SEC to release its full report on the Harken investigation . When the Rangers
franchise was sold for $250 million in 1998 , at a total profit of $170 million , Bush personally received $14.9 million for his $600,000 investment . As
managing general partner of the Rangers , Bush assisted the team 's media relations and the construction of a new stadium . Many in Arlington protested that the
stadium was paid for with public funds , and when the stadium title was given to the Texas Rangers Organization , it effectively allowed Bush to cash out public
funds . His public role generated valuable goodwill and reinforced name recognition throughout Texas .
</item 2>

<item 3>
Professional life of George W. Bush
Professional life of George W. Bush - wikipedia Professional life of George W. Bush Jump to : navigation , search George W. Bush This article covers the
professional life of George W. Bush , the 43rd President of the United States . Prior to his election as President in 2000 , Bush held numerous other positions
, including being an oil executive , an owner of the Texas Rangers baseball team , and Governor of Texas . ( Also see Governorship of George W. Bush . )
Contents ( hide ) 1 Business 1.1 Texas Rangers and allegations of insider trading 1.2 Early political career 2 See also 3 References Business ( edit ) Bush
began his industry career in 1979 , when he established Arbusto Energy , an oil and gas exploration company he financed with his education trust fund surplus
and money from other investors , including Dorothy Bush , Lewis Lehrman , William Henry Draper III , Bill Gammell , and James R. Bath , the last of whom
represented Salem bin Laden , a half - brother and cousin of Osama bin Laden .
</item 3>

<item 4>
George W. Bush founded oil firm Arbusto Energy in 1979, sold it to Spectrum 7 in 1984, and joined Harken Energy after its 1986 merger. In 1989, he invested
about $606,000 for a 2% stake in the Texas Rangers; the franchise’s 1998 sale earned him $14.9 million. His 1990 Harken stock sale, shortly before major losses
were reported, prompted insider-trading allegations.
</item 4>

<item 5>
Houston’s professional baseball team was the minor-league Buffaloes from 1888 to 1961. George Kirksey, Craig Cullinan, oilman R.E. “Bob” Smith, and former mayor
and judge Roy Hofheinz formed the Houston Sports Association to secure a major-league team. After MLB resisted expansion, they helped launch the rival
Continental League, prompting the established leagues to expand. The Continental League folded in 1960; on October 17, the National League awarded Houston
</item 5>

<item 6>
Professional life of George W. Bush
Against the advice of his counsel , Bush repaid the loan by selling $848,000 worth of stock in Harken Energy . Harken reported significant financial losses
within a year of this sale , triggering allegations of insider trading . On March 27 , 1992 , the Securities and Exchange Commission concluded that Bush had a
`` preexisting plan '' to sell , that Bush had a `` relatively limited role in Harken management '' , and that it had not seen evidence of insider trading . The
subsequent SEC investigation ended in 1992 with a memo stating `` it appears that Bush did not engage in illegal insider trading , '' but noted that the memo ``
must in no way be construed as indicating that the party has been exonerated or that no action may ultimately result '' . Critics allege that this decision was
strongly influenced by the makeup of the SEC at the time , which heavily favored Bush . The chairman at the time was Richard Breeden , a good friend of the Bush
family 's who had been nominated to the SEC by President George H.W. Bush and who had been a lawyer in James Baker 's firm , Baker Botts .
</item 6>

<item 7>
Houston Astros
In November 2010 , Drayton McLane announced that the Astros were being put up for sale . McLane stated that because the Astros were one of the few franchises in
Major League Baseball with only one family as the owners , he was planning his estate . McLane was 75 years old as of November 2011 . In March 2011 , local
Houston businessman Jim Crane emerged as the front - runner to purchase the franchise . In the 1980s , Crane founded an air freight business which later merged
with CEVA Logistics , and later founded Crane Capital Group . McLane and Crane had a previous handshake agreement for the franchise in 2008 , but Crane abruptly
changed his mind and broke off discussions . Crane also attempted to buy the Chicago Cubs in 2008 and the Texas Rangers during their 2010 bankruptcy auction .
Crane came under scrutiny because of previous allegations of discriminatory hiring practices regarding women and minorities , among other issues . This delayed
MLB 's approval process . During the summer of 2011 , a frustrated Crane hinted that the delays might threaten the deal .
</item 7>

<item 8>
Professional life of George W. Bush
In 1984 , Bush sold the company , hurt in the wake of the 1979 energy crisis , to Spectrum 7 , another Texas gas exploration firm . Under the terms of the sale
, Bush became CEO . Spectrum 7 lost revenue and was merged into Harken Energy Corporation in 1986 , with Bush becoming a director of Harken . Texas Rangers and
allegations of insider trading ( edit ) After working on his father 's successful 1988 presidential campaign , Bush learned from fellow Yale alumnus William
DeWitt , Jr. , that family friend Eddie Chiles wanted to sell the Texas Rangers baseball franchise along with the new sports stadium ; built on land acquired
under eminent domain law and built under funding financed through taxpayers ' funds backed by a bond issued for its debt . In April 1989 , Bush assembled a
group of investors from his father 's close friends , including fellow fraternity brother Roland W. Betts ; the group bought an 86 % share of the Rangers for
$75 million . Bush received a 2 % share by investing $606,302 , of which $500,000 was a bank loan .
</item 8>

<item 9>
Professional life of George W. Bush
In House of Bush , House of Saud , Craig Unger notes that at the time of Bush 's sale , Harken Energy `` was expected to run out of money in just three days ''
( p. 123 ) . In a last - ditch attempt to save the company , Harken was advised by the endowment fund of Harvard University to spin off two of its lower -
performing divisions . `` According to a Harken memo , if the plan did not go through , the company had ' no other source of immediate financing . ' '' Bush had
already taken out a $500,000 loan and sought Harken 's general counsel for advice . The reply was explicit : `` The act of trading , particularly if close in
time to the receipt of the inside information , is strong evidence that the insider 's investment decision was based on the inside information ... the insider
should be advised not to sell '' . This memo was turned over by Bush 's attorney the day after the U.S. Securities and Exchange Commission ( SEC ) ruled that it
would not charge Bush with insider trading .
</item 9>

<item 10>
Professional life of George W. Bush
Jump up ^ `` CNN.com - White House defends Bush handling of stock sale - July 3 , 2002 '' . Jump up ^ `` Files : Bush Knew Firm 's Plight Before Stock Sale '' .
Washington Post . 2002 - 07 - 21 . Retrieved 2007 - 01 - 02 . Jump up ^ http://www.austinchronicle.com/issues/vol17/issue19/pols.bush.html Jump up ^ `` Team
Player - Texas - Joe Nick Patoski '' . Jump up ^ `` ESPN.com - Major League Baseball - Friday : Bush gets to know Texas '' . Jump up ^ `` Bush Was n't Always a
Front - Runner '' . Washington Post . 1999 - 10 - 17 . Retrieved 2007 - 04 - 07 .
</item 10>
=== END ITEM id=be2eec9aa07b11cc ===

=== ITEM id=4a607ffecc843a4d ===
Gold sentences and facts:
Sentence 0: Gina Tognoni / toʊnˈjoʊni / ( born November 28 , 1973 ) is an American actress , best known for her work with American daytime soap operas . Her most notable performances include Kelly Cramer on One Life to Live and Dinah Marler on Guiding Light . She is currently starring as Phyllis Summers on The Young and the Restless .
  Fact 0.0: Gina Tognoni was born on November 28, 1973.
  Fact 0.1: Gina Tognoni is an American actress.
  Fact 0.2: Gina Tognoni is best known for her work with American daytime soap operas.
  Fact 0.3: Gina Tognoni's most notable performances include Kelly Cramer on One Life to Live.
  Fact 0.4: Gina Tognoni's most notable performances include Dinah Marler on Guiding Light.
  Fact 0.5: Gina Tognoni is currently starring as Phyllis Summers on The Young and the Restless.
Sentence 1: Gina Tognoni ( 1973 - 11 - 28 ) November 28 , 1973 ( age 44 ) St. Louis , Missouri , U.S. Occupation Actress businesswoman Years active 1995 -- present Television Guiding Light ( 2004 -- 2009 ) One Life to Live ( 1995 -- 2001 , 2002 , 2010 -- 2011 ) The Young and the Restless ( 2014 -- present ) Spouse ( s ) Joseph Chiarello ( m . 2009 )
  Fact 1.0: Gina Tognoni was born on November 28, 1973.
  Fact 1.1: Gina Tognoni is age 44.
  Fact 1.2: Gina Tognoni was born in St. Louis, Missouri, U.S.
  Fact 1.3: Gina Tognoni's occupations are actress and businesswoman.
  Fact 1.4: Gina Tognoni's years active are 1995 to present.
  Fact 1.5: Gina Tognoni was on Guiding Light from 2004 to 2009.
  Fact 1.6: Gina Tognoni was on One Life to Live from 1995 to 2001, in 2002, and from 2010 to 2011.
  Fact 1.7: Gina Tognoni was on The Young and the Restless from 2014 to present.
  Fact 1.8: Gina Tognoni's spouse is Joseph Chiarello.
  Fact 1.9: Gina Tognoni married Joseph Chiarello in 2009.

Retrieved items:
<item 1>
Sheila Carter
( 1991 -- 92 ) Eric Forrester ( 1993 -- 95 ) James Warwick ( 1997 -- 98 ) Sons Ryder Callahan Daughters Mary Carter Warwick Diana Carter Daisy Carter
Granddaughters Lucy Romalotti Sheila Carter is a fictional character from The Young and the Restless and The Bold and the Beautiful , American soap operas on
the CBS network . Created by William J. Bell , the role was introduced in 1990 -- by Edward J. Scott -- under the portrayal of Kimberlin Brown , who portrayed
the role for many intervals on both soaps until 2006 . That same year , Michelle Stafford took over the role , after Sheila had plastic surgery to look like
Phyllis Summers . Brown returned to the role of Sheila on The Bold and the Beautiful , beginning on June 9 , 2017 . Sheila is known as a villain . A significant
portion of Sheila 's history on both soaps revolves around her long - running conflict with Lauren Fenmore , whom she has attempted to kill on multiple
occasions .
</item 1>

<item 2>
Sheila Carter
CBS.com . Retrieved 16 September 2012 . Jump up ^ `` The Young and the Restless : Wednesday , 11th January 2006 '' . CBS.com . Retrieved 16 September 2012 .
Jump up ^ `` The Young and the Restless : Thursday , 28th December 2006 '' . CBS.com . Retrieved 16 September 2012 . Jump up ^ `` Michelle Stafford Playing
Sheila Carter on The Young and the Restless '' . She Knows Soaps . Retrieved 16 September 2012 . Jump up ^ Kimberlin Brown ( 2007 ) . KB Interview Nov 2007 .
KOUS News . Retrieved 16 September 2012 . Jump up ^ `` The Young and the Restless : Friday , 9th February 2007 '' . CBS.com . Retrieved 16 September 2012 . Jump
up ^ Kerr , Luke ( 28 August 2009 ) . `` Could Kimberlin Brown Return to B&B ? '' . Daytime Confidential . Retrieved 16 September 2012 .
</item 2>

<item 3>
Gina Tognoni
She is currently starring as Phyllis Summers on The Young and the Restless . Contents ( hide ) 1 Career 2 Personal life 3 Filmography 4 Awards and nominations 5
References 6 External links Career ( edit ) Tognoni is known for her roles on daytime television soap operas . In 1995 , she made her soap debut as Kelly Cramer
in ABC 's One Life to Live . She appeared on the show until 2001 . In April 2001 , it was announced that Tognoni , after six years with the show , had decided
not to renew her contract with the series . Tognoni returned to daytime television as Dinah Marler on CBS 's Guiding Light in July 2004 , and won her first
Daytime Emmy Award for Outstanding Supporting Actress in a Drama Series in 2006 for her work on the show . In 2007 , she was nominated again in the same
category but lost to Genie Francis . She won her second Emmy in 2008 , becoming only the second actress to be a repeat winner in the Supporting Actress category
.
</item 3>

<item 4>
Gina Tognoni
. Soap Opera Digest . American Media , Inc . Retrieved May 22 , 2014 . Jump up ^ Logan , Michael ( May 22 , 2014 ) . `` Exclusive : Gina Tognoni Joins The Young
and the Restless '' . TV Guide . CBS Interactive . Retrieved May 22 , 2014 . Jump up ^ `` Gina Tognoni : Biography '' . TV Guide . Retrieved 7 October 2014 .
Jump up ^ `` Ellen DeGeneres repeats sweep of Daytime Emmys '' . Kentucky New Era . Los Angeles : Taylor W. Hayes . Associated Press . May 1 , 2006 . p. 8 .
Retrieved May 6 , 2013 . Jump up ^ Bonawitz , Amy ( February 11 , 2009 ) . `` 34th Annual Daytime Emmy Winners '' . New York City : CBS News . Archived from the
original on June 28 , 2013 . Retrieved January 8 , 2013 . Jump up ^ Silverman , Stephen M.
</item 4>

<item 5>
Sheila Carter
However , that December , the character returned , though not portrayed by Brown , but by Michelle Stafford , who in addition portrayed Phyllis Summers . For
the reason of Brown not returning to the role , the actress said : `` I had a bad accident last year , and when they wanted me back in November , I really could
n't go at that time , and that 's why I came back looking like somebody else . '' Stafford only portrayed the role for two months until February 9 , 2007 , when
Sheila was shot by her longtime enemy Lauren Fenmore and presumed dead . In 2009 , reports speculated that Brown would reprise the role again on The Bold and
the Beautiful , which was followed by more rumors speculating a return to The Young and the Restless . `` I actually got a call from ( B&B head writer and
executive producer ) Brad Bell a while back asking if I 'd be interested in coming back , and I said , ' Sure ! ' The thing with daytime is that the storylines
are ( written ) so far in advance , you just never know when ( you might be needed ) , '' Brown said at the time .
</item 5>

<item 6>
Gina Tognoni
Gina Tognoni - wikipedia Gina Tognoni Jump to : navigation , search Gina Tognoni ( 1973 - 11 - 28 ) November 28 , 1973 ( age 44 ) St. Louis , Missouri , U.S.
Occupation Actress businesswoman Years active 1995 -- present Television Guiding Light ( 2004 -- 2009 ) One Life to Live ( 1995 -- 2001 , 2002 , 2010 -- 2011 )
The Young and the Restless ( 2014 -- present ) Spouse ( s ) Joseph Chiarello ( m . 2009 ) Gina Tognoni / toʊnˈjoʊni / ( born November 28 , 1973 ) is an American
actress , best known for her work with American daytime soap operas . Her most notable performances include Kelly Cramer on One Life to Live and Dinah Marler on
Guiding Light . She is currently starring as Phyllis Summers on The Young and the Restless .
</item 6>

<item 7>
Sheila Carter
Despite being a regular cast member on The Bold and the Beautiful , Brown returned to The Young and the Restless for several guest appearances beginning on
October 27 , 1992 . Brown departed on November 5 , 1992 , and reappeared for one episode on December 23 , 1992 . She continued making appearance from January 7
to 14 , 1993 , February 19 , 1993 , to March 2 , 1993 , October 11 , 1993 , October 19 to 27 , 1993 , January 12 , 1994 , and February 28 , 1995 , until March 2
, 1995 . In July 2005 , after a 10 - year absence from The Young and the Restless , it was announced that Brown was to reprise the role . At the time , Brown
had been appearing on ABC 's One Life to Live as Paige Miller , and Brown reportedly opted to leave the role when ABC was unable to match the financial terms of
a contract offered to her by CBS . She returned on August 5 , 2005 , and her run ended on January 11 , 2006 .
</item 7>

<item 8>
Graph query row:
person: Gina Tognoni
birth_date: November 28, 1973
work: Exclusive: Gina Tognoni Joins The Young and the Restless
role: None
</item 8>

<item 9>
Gina Tognoni
In 2009 , after CBS canceled Guiding Light , Tognoni returned to her role as Kelly Cramer on One Life to Live . Also it was reported that Tognoni was in talks
with several soaps for her post-Guiding Light gig . Following the April 2011 cancelation of One Life to Live and All My Children , it was announced that Tognoni
and co-star Tom Degnan , who was portraying Joey Buchanan at the time , would exit the series before its January 2012 finale . In addition to her career on
soaps , Tognoni had guest - starring roles in Law & Order : Special Victims Unit and The Sopranos , and co-starred in a number of television shows and
independent films . On May 22 , 2014 , it was announced that Tognoni was cast as Phyllis Summers in CBS 's The Young and the Restless , replacing Michelle
Stafford . Tognoni made her debut on August 11 , 2014 . Personal Life ( edit ) Tognoni was born in St. Louis , Missouri .
</item 9>

<item 10>
Gina Tognoni
( 2009 - 06 - 08 ) . `` Gina Tognoni close to deciding on her post-GL career '' . SoapCentral . Retrieved June 28 , 2012 . Jump up ^ `` AMC and OLTL Canceled ''
. Soap Opera Digest . Soapoperadigest.com. 2011 - 04 - 14 . Retrieved June 28 , 2012 . Jump up ^ Kroll , Dan J. ( 2011 - 06 - 07 ) . `` Gina Tognoni , Tom
Degnan exiting OLTL this summer '' . SoapCentral . Retrieved June 28 , 2012 . Jump up ^ Fairman , Michael ( 2011 - 05 - 31 ) . `` Is Gina Tognoni headed out of
One Life to Live early ? '' . Michael Fairman On - Air On - Soaps . Retrieved June 28 , 2012 . Jump up ^ SOD ( May 22 , 2014 ) . `` GINA TOGNONI IS Y&R 'S NEW
PHYLLIS ! '' . Soap Opera Digest .
</item 10>
=== END ITEM id=4a607ffecc843a4d ===
