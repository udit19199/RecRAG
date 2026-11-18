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

=== ITEM id=d11e373cba75e7cc ===
Gold sentences and facts:
Sentence 0: In healthy adults , there are two normal heart sounds , often described as a lub and a dub ( or dup ) , that occur in sequence with each heartbeat . These are the first heart sound ( S ) and second heart sound ( S ) , produced by the closing of the atrioventricular valves and semilunar valves , respectively . In addition to these normal sounds , a variety of other sounds may be present including heart murmurs , adventitious sounds , and gallop rhythms S and S .
  Fact 0.0: In healthy adults there are two normal heart sounds.
  Fact 0.1: The two normal heart sounds are often described as a lub and a dub (or dup).
  Fact 0.2: The two normal heart sounds occur in sequence with each heartbeat.
  Fact 0.3: The first heart sound (S) is produced by the closing of the atrioventricular valves.
  Fact 0.4: The second heart sound (S) is produced by the closing of the semilunar valves.
  Fact 0.5: The closing of the atrioventricular valves and the semilunar valves produces the first and second heart sounds, respectively.
  Fact 0.6: In addition to the normal sounds, a variety of other sounds may be present.
  Fact 0.7: The other sounds that may be present include heart murmurs, adventitious sounds, and gallop rhythms labeled S and S.

Retrieved items:
<item 1>
Heart sounds
Aortic blood flow quickly reverses back toward the left ventricle , catching the pocket - like cusps of the aortic valve , and is stopped by aortic valve
closure . Similarly , as the pressure in the right ventricle falls below the pressure in the pulmonary artery , the pulmonary valve closes . The S sound results
from reverberation within the blood associated with the sudden block of flow reversal . Splitting of S2 , also known as physiological split , normally occurs
during inhalation because the decrease in intrathoracic pressure increases the time needed for pulmonary pressure to exceed that of the right ventricular
pressure . A widely split S2 can be associated with several different cardiovascular conditions , including left bundle branch block , pulmonary stenosis , and
atrial septal defect . Extra heart sounds ( edit ) The rarer extra heart sounds form gallop rhythms and are heard in both normal and abnormal situations .
</item 1>

<item 2>
Heart sounds
S3 can be a normal finding in young patients but is generally pathologic over the age of 40 . The most common cause of pathologic S3 is congestive heart failure
. Main article : Fourth heart sound S4 when audible in an adult is called a presystolic gallop or atrial gallop . This gallop is produced by the sound of blood
being forced into a stiff or hypertrophic ventricle . `` ta - lub - dub '' or `` a-stiff - wall '' It is a sign of a pathologic state , usually a failing or
hypertrophic left ventricle , as in systemic hypertension , severe valvular aortic stenosis , and hypertrophic cardiomyopathy . The sound occurs just after
atrial contraction at the end of diastole and immediately before S1 , producing a rhythm sometimes referred to as the `` Tennessee '' gallop where S4 represents
the `` Ten - '' syllable . It is best heard at the cardiac apex with the patient in the left lateral decubitus position and holding his breath . The combined
presence of S3 and S4 is a quadruple gallop , also known as the `` Hello - Goodbye '' gallop .
</item 2>

<item 3>
Heart sounds
The reason the third heart sound does not occur until the middle third of diastole is probably that during the early part of diastole , the ventricles are not
filled sufficiently to create enough tension for reverberation . It may also be a result of tensing of the chordae tendineae during rapid filling and expansion
of the ventricle . In other words , an S3 heart sound indicates increased volume of blood within the ventricle . An S3 heart sound is best heard with the bell -
side of the stethoscope ( used for lower frequency sounds ) . A left - sided S3 is best heard in the left lateral decubitus position and at the apex of the
heart , which is normally located in the 5th left intercostal space at the midclavicular line . A right - sided S3 is best heard at the lower - left sternal
border . The way to distinguish between a left and right - sided S3 is to observe whether it increases in intensity with inhalation or exhalation . A right -
sided S3 will increase on inhalation , while a left - sided S3 will increase on exhalation .
</item 3>

<item 4>
Heart sounds
Main article : Third heart sound Rarely , there may be a third heart sound also called a protodiastolic gallop , ventricular gallop , or informally the ``
Kentucky '' gallop as an onomatopoeic reference to the rhythm and stress of S1 followed by S2 and S3 together ( S1 = Ken ; S2 = tuck ; S3 = y ) . `` lub - dub -
ta '' or `` slosh - ing - in '' If new , indicates heart failure or volume overload . It occurs at the beginning of diastole after S2 and is lower in pitch than
S1 or S2 as it is not of valvular origin . The third heart sound is benign in youth , some trained athletes , and sometimes in pregnancy but if it re-emerges
later in life it may signal cardiac problems , such as a failing left ventricle as in dilated congestive heart failure ( CHF ) . S3 is thought to be caused by
the oscillation of blood back and forth between the walls of the ventricles initiated by blood rushing in from the atria .
</item 4>

<item 5>
Heart sounds
L.V. Left ventricle . P.A. Pulmonary artery . R.A. Right atrium . R.V. Right ventricle . V.S. Ventricular septum . Heart sounds are the noises generated by the
beating heart and the resultant flow of blood through it . Specifically , the sounds reflect the turbulence created when the heart valves snap shut . In cardiac
auscultation , an examiner may use a stethoscope to listen for these unique and distinct sounds that provide important auditory data regarding the condition of
the heart . In healthy adults , there are two normal heart sounds , often described as a lub and a dub ( or dup ) , that occur in sequence with each heartbeat .
These are the first heart sound ( S ) and second heart sound ( S ) , produced by the closing of the atrioventricular valves and semilunar valves , respectively
. In addition to these normal sounds , a variety of other sounds may be present including heart murmurs , adventitious sounds , and gallop rhythms S and S .
Heart murmurs are generated by turbulent flow of blood , which may occur inside or outside the heart .
</item 5>

<item 6>
Heart sounds
This is a characteristic scratching , creaking , high - pitched sound emanating from the rubbing of both layers of inflamed pericardium . It is the loudest in
systole , but can often be heard at the beginning and at the end of diastole . It is very dependent on body position and breathing , and changes from hour to
hour . Surface anatomy ( edit ) The aortic area , pulmonic area , tricuspid area and mitral area are areas on the surface of the chest where the heart is
auscultated . Heart sounds result from reverberation within the blood associated with the sudden block of flow reversal by the valves closing . Because of this
, auscultation to determine function of a valve is usually not performed at the position of the valve , but at the position to where the sound waves reverberate
.
</item 6>

<item 7>
Heart sounds
Aortic valve ( to aorta ) right second intercostal space upper right sternal border Pulmonary valve ( to pulmonary trunk ) left second intercostal space upper
left sternal border Erb 's point Left third intercostal space left sternal border Tricuspid valve ( to right ventricle ) left fourth , fifth intercostal spaces
lower left sternal border Mitral valve ( to left ventricle ) left fifth intercostal space left midclavicular line Recording heart sounds ( edit ) Using
electronic stethoscopes , it is possible to record heart sounds via direct output to an external recording device , such as a laptop or MP3 recorder . The same
connection can be used to listen to the previously recorded auscultation through the stethoscope headphones , allowing for more detailed study of murmurs and
other heart sounds , for general research as well as evaluation of a particular patient 's condition .
</item 7>

<item 8>
Heart sounds
Heart sounds - wikipedia Heart sounds Jump to : navigation , search Normal heart sounds Normal heart sounds as heard with a stethoscope Problems playing this
file ? See media help . 72 BPM Heart sound Front of thorax , showing surface relations of bones , lungs ( purple ) , pleura ( blue ) , and heart ( red outline )
. The locations of best auscultation for each heart valve are labeled with `` M '' , `` T '' , `` A '' , and `` P '' . First heart sound : caused by
atrioventricular valves - Mitral ( M ) and Tricuspid ( T ) . Second heart sound caused by semilunar valves -- Aortic ( A ) and Pulmonary / Pulmonic ( P ) .
Diagram showing relations of opened heart to front of thoracic wall . Ant . Anterior segment of tricuspid valve . A O. Aorta . A.P. Anterior papillary muscle .
In . Innominate artery . L.C.C. Left common carotid artery . L.S. Left subclavian artery . L.V. Left ventricle . P.A. Pulmonary artery .
</item 8>

<item 9>
Heart sounds
The sudden tensing of the chordae tendineae and the squeezing of the ventricles against closed semilunar valves , sends blood rushing back toward the atria ,
and the parachute - like valves catch the rush of blood in their leaflets causing the valve to snap shut . The S1 sound results from reverberation within the
blood associated with the sudden block of flow reversal by the valves . The delay of T1 even more than normally causes the split S1 which is heard in a right
bundle branch blockage . The second heart sound , or S , forms the `` dub '' of `` lub - dub '' and is composed of components A ( aortic valve closure ) and P (
pulmonary valve closure ) . Normally A precedes P especially during inspiration where a split of S can be heard . It is caused by the closure of the semilunar
valves ( the aortic valve and pulmonary valve ) at the end of ventricular systole and the beginning of ventricular diastole . As the left ventricle empties ,
its pressure falls below the pressure in the aorta .
</item 9>

<item 10>
Heart sounds
See also ( edit ) Pulse Physical examination Precordial examination Benign pediatric heart murmur Iambic pentameter , a metre in poetry which follows a similar
rhythm to the human heartbeat Pulsatile tinnitus - hearing a heartbeat sound in one or both ears Notes and references ( edit ) Jump up ^
https://depts.washington.edu/physdx/heart/tech2.html Jump up ^ https://depts.washington.edu/physdx/heart/tech2.html Jump up ^ `` thrill '' . Medline Plus
Medical Dictionary . Jump up ^ Bernstein , Lenny ( 2016 - 01 - 02 ) , `` Heart doctors are listening for clues to the future of their stethoscopes '' ,
Washington Post . External links ( edit ) University of Michigan Heart Sound and Murmur Library . Auscultation Assistant . UCLA Heart Sounds - Heart Murmurs .
practicalclinicalskills.com Physiology of the cardiovascular system Heart Cardiac output Cardiac cycle Cardiac output Heart rate Stroke volume Stroke volume End
- diastolic volume End - systolic volume Afterload Preload Frank -- Starling law Cardiac function curve Venous return curve Wiggers diagram Pressure volume
diagram Ultrasound Fractional
</item 10>
=== END ITEM id=d11e373cba75e7cc ===

=== ITEM id=cc5cb7cd4fa14a20 ===
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
She has also had conflicts with Stephanie Forrester , Maggie Forrester , Amber Moore , Taylor Hayes , Brooke Logan , Phyllis Summers and Quinn Fuller . Contents
( hide ) 1 Casting and creation 2 Development 3 Storylines 3.1 1990 -- 92 3.2 1992 -- 98 3.3 2002 -- 03 3.4 2005 -- 07 3.5 2010 , 2012 3.6 2017 -- 4 Critical
reception 5 References 6 External links Casting and creation ( edit ) The role was portrayed by Kimberlin Brown for a total of 16 years . Her first run was on
The Young and the Restless from May 1990 to May 20 , 1992 , when the character crossed over to The Bold and the Beautiful from May 21 , 1992 , to October 2 ,
1998 , again from May 24 to November 5 , 2002 , and again finally from September 26 to October 10 , 2003 .
</item 1>

<item 2>
Sheila Carter
Sheila continues to insinuate herself into Eric 's life , faking a concussion after a catfight with Quinn to manipulate Eric into letting her stay in the
Forrester mansion . Sheila blackmails her ex-husband , James Warwick , into lying about the severity of her condition . Critical reception ( edit ) Sheila is
often considered to be one of the most popular villains in daytime . Kimberlin Brown was nominated for a Daytime Emmy in the role as Outstanding Supporting
Actress in 1993 . With the global success of The Bold and the Beautiful , Sheila is arguably the most recognisable soap opera villain in the world . References
( edit ) ^ Jump up to : `` About Y&R : Who 's Who in Los Angeles - Sheila Carter '' . SoapCental . Retrieved 16 September 2012 . ^ Jump up to : Kroll , Dan J. (
9 July 2005 ) . `` Y&R confirms the return of Kimberlin Brown '' . SoapCentral . Retrieved 16 September 2012 . Jump up ^ `` The Young and the Restless : Friday
, 5th August 2005 '' . CBS.com .
</item 2>

<item 3>
Sheila Carter
However , that December , the character returned , though not portrayed by Brown , but by Michelle Stafford , who in addition portrayed Phyllis Summers . For
the reason of Brown not returning to the role , the actress said : `` I had a bad accident last year , and when they wanted me back in November , I really could
n't go at that time , and that 's why I came back looking like somebody else . '' Stafford only portrayed the role for two months until February 9 , 2007 , when
Sheila was shot by her longtime enemy Lauren Fenmore and presumed dead . In 2009 , reports speculated that Brown would reprise the role again on The Bold and
the Beautiful , which was followed by more rumors speculating a return to The Young and the Restless . `` I actually got a call from ( B&B head writer and
executive producer ) Brad Bell a while back asking if I 'd be interested in coming back , and I said , ' Sure ! ' The thing with daytime is that the storylines
are ( written ) so far in advance , you just never know when ( you might be needed ) , '' Brown said at the time .
</item 3>

<item 4>
Sheila Carter
Despite being a regular cast member on The Bold and the Beautiful , Brown returned to The Young and the Restless for several guest appearances beginning on
October 27 , 1992 . Brown departed on November 5 , 1992 , and reappeared for one episode on December 23 , 1992 . She continued making appearance from January 7
to 14 , 1993 , February 19 , 1993 , to March 2 , 1993 , October 11 , 1993 , October 19 to 27 , 1993 , January 12 , 1994 , and February 28 , 1995 , until March 2
, 1995 . In July 2005 , after a 10 - year absence from The Young and the Restless , it was announced that Brown was to reprise the role . At the time , Brown
had been appearing on ABC 's One Life to Live as Paige Miller , and Brown reportedly opted to leave the role when ABC was unable to match the financial terms of
a contract offered to her by CBS . She returned on August 5 , 2005 , and her run ended on January 11 , 2006 .
</item 4>

<item 5>
Sheila Carter
Sheila Carter - Wikipedia Sheila Carter Jump to : navigation , search Sheila Carter Kimberlin Brown as Sheila Carter The Young and the Restless character
Portrayed by Kimberlin Brown ( 1990 -- ) Michelle Stafford ( 2006 -- 07 ) Duration 1990 -- 98 2002 -- 03 2005 -- 07 2017 -- First appearance May 1990 Created by
William J. Bell Introduced by Edward J. Scott ( 1990 , 1992 -- 95 ) Lee Phillip Bell ( 1992 ) Bradley Bell ( 2002 -- 03 ) John F. Smith ( 2005 ) Lynn Marie
Latham and Josh Griffith ( 2006 ) Bradley Bell and Michael Minnis ( 2017 ) Crossover appearances The Bold and the Beautiful Classification Present ; regular
Profile Other names Sheila Grainger Sheila Forrester Sheila Warwick Sybil Weller Sheila Lovejoy Brenda Harris Jennifer Mitchell Phyllis Newman Betsy Occupation
Waitress Executive assistant Nurse Residence Los Angeles , California Michelle Stafford as Sheila Carter ( show ) Family Family Carter Mother Molly Carter
Sisters Sarah Smythe Husband Scott Grainger , Sr.
</item 5>

<item 6>
Gina Tognoni
She is currently starring as Phyllis Summers on The Young and the Restless . Contents ( hide ) 1 Career 2 Personal life 3 Filmography 4 Awards and nominations 5
References 6 External links Career ( edit ) Tognoni is known for her roles on daytime television soap operas . In 1995 , she made her soap debut as Kelly Cramer
in ABC 's One Life to Live . She appeared on the show until 2001 . In April 2001 , it was announced that Tognoni , after six years with the show , had decided
not to renew her contract with the series . Tognoni returned to daytime television as Dinah Marler on CBS 's Guiding Light in July 2004 , and won her first
Daytime Emmy Award for Outstanding Supporting Actress in a Drama Series in 2006 for her work on the show . In 2007 , she was nominated again in the same
category but lost to Genie Francis . She won her second Emmy in 2008 , becoming only the second actress to be a repeat winner in the Supporting Actress category
.
</item 6>

<item 7>
Gina Tognoni
Gina Tognoni - wikipedia Gina Tognoni Jump to : navigation , search Gina Tognoni ( 1973 - 11 - 28 ) November 28 , 1973 ( age 44 ) St. Louis , Missouri , U.S.
Occupation Actress businesswoman Years active 1995 -- present Television Guiding Light ( 2004 -- 2009 ) One Life to Live ( 1995 -- 2001 , 2002 , 2010 -- 2011 )
The Young and the Restless ( 2014 -- present ) Spouse ( s ) Joseph Chiarello ( m . 2009 ) Gina Tognoni / toʊnˈjoʊni / ( born November 28 , 1973 ) is an American
actress , best known for her work with American daytime soap operas . Her most notable performances include Kelly Cramer on One Life to Live and Dinah Marler on
Guiding Light . She is currently starring as Phyllis Summers on The Young and the Restless .
</item 7>

<item 8>
Sheila Carter
CBS.com . Retrieved 16 September 2012 . Jump up ^ `` The Young and the Restless : Wednesday , 11th January 2006 '' . CBS.com . Retrieved 16 September 2012 .
Jump up ^ `` The Young and the Restless : Thursday , 28th December 2006 '' . CBS.com . Retrieved 16 September 2012 . Jump up ^ `` Michelle Stafford Playing
Sheila Carter on The Young and the Restless '' . She Knows Soaps . Retrieved 16 September 2012 . Jump up ^ Kimberlin Brown ( 2007 ) . KB Interview Nov 2007 .
KOUS News . Retrieved 16 September 2012 . Jump up ^ `` The Young and the Restless : Friday , 9th February 2007 '' . CBS.com . Retrieved 16 September 2012 . Jump
up ^ Kerr , Luke ( 28 August 2009 ) . `` Could Kimberlin Brown Return to B&B ? '' . Daytime Confidential . Retrieved 16 September 2012 .
</item 8>

<item 9>
Sheila Carter
( 1991 -- 92 ) Eric Forrester ( 1993 -- 95 ) James Warwick ( 1997 -- 98 ) Sons Ryder Callahan Daughters Mary Carter Warwick Diana Carter Daisy Carter
Granddaughters Lucy Romalotti Sheila Carter is a fictional character from The Young and the Restless and The Bold and the Beautiful , American soap operas on
the CBS network . Created by William J. Bell , the role was introduced in 1990 -- by Edward J. Scott -- under the portrayal of Kimberlin Brown , who portrayed
the role for many intervals on both soaps until 2006 . That same year , Michelle Stafford took over the role , after Sheila had plastic surgery to look like
Phyllis Summers . Brown returned to the role of Sheila on The Bold and the Beautiful , beginning on June 9 , 2017 . Sheila is known as a villain . A significant
portion of Sheila 's history on both soaps revolves around her long - running conflict with Lauren Fenmore , whom she has attempted to kill on multiple
occasions .
</item 9>

<item 10>
Sheila Carter
She then uses crutches to walk , has a seat and asks him for a new face . Months later : it is revealed that Sheila Carter is still alive , had plastic surgery
to look like Phyllis Summers Newman ( Michelle Stafford ) , and that Paul Williams had been hiding the fact that he had been keeping her locked in a cage she
had constructed in an abandoned warehouse . Ultimately , Sheila is determined to escape and take Phyllis 's place in order to infiltrate Michael and Lauren 's
lives . Sheila is accidentally freed from the prison she created for Lauren by Maggie Sullivan , Paul 's girlfriend . After acting as a kidnap victim and
pleading for Maggie 's help , Maggie sets her free but Sheila chokes her unconscious , steals her gun , and then shoots her , but Maggie survives . Subsequently
, Paul returns to the warehouse only to be ambushed by Sheila who imprisons both Paul and Maggie in that cage . Sheila then went to Phyllis ' , tied her up and
waited for Lauren and baby Fen to arrive while taunting Phyllis about how great it would be to take over her life with her husband Nicholas Newman ( Joshua
Morrow ) and her baby Summer .
</item 10>
=== END ITEM id=cc5cb7cd4fa14a20 ===

=== ITEM id=b6e3c760019d123b ===
Gold sentences and facts:
Sentence 0: In most of the world speed limit signs display the limitation within a red circle . This design follows the style set out by the Vienna Convention on Road Signs and Signals , with the exception of United States and Canada . All countries globally worldwide ( except the United Kingdom and United States ) use metric speed limits in kilometres per hour . Metric signs are optional in the United States though less commonly seen , while Samoa uses both units . In Ontario , Canada , the type , location , and frequency of speed limit signs is covered by regulation 615 of the Ontario Highway Traffic Act .
  Fact 0.0: In most of the world, speed limit signs display the limitation within a red circle.
  Fact 0.1: The red-circle speed limit sign design follows the style set out by the Vienna Convention on Road Signs and Signals.
  Fact 0.2: The Vienna Convention style is not followed by the United States and Canada.
  Fact 0.3: All countries worldwide, except the United Kingdom and the United States, use metric speed limits in kilometres per hour.
  Fact 0.4: Metric speed limit signs are optional in the United States.
  Fact 0.5: Metric speed limit signs are less commonly seen in the United States.
  Fact 0.6: Samoa uses both units for speed limits.
  Fact 0.7: In Ontario, Canada, the type, location, and frequency of speed limit signs is covered by regulation 615 of the Ontario Highway Traffic Act.

Retrieved items:
<item 1>
Speed limit
The U.S. Federal Highway Administration 's Manual on Uniform Traffic Control Devices provides guidelines for the appearance of speed limit signs . In Alaska and
California , speed limits are often labeled `` MAXIMUM SPEED '' instead . In Oregon , most speed limit signs just say `` SPEED '' and the number . Canada has
similar signs bearing the legend `` MAXIMUM '' and in km / h instead of mph . `` Maximum '' is used instead of `` Speed Limit '' because it has similar meanings
in English and French . Australian signs are rectangular but have a red circle like the Vienna Convention signs . The speed limit is commonly set at or below
the 85th percentile operating speed ( being the speed which no more than 15 % of traffic is exceeding ) and in the US is frequently set 4 to 8 mph ( 6 to 13 km
/ h ) below that speed . Thus , if the 85th percentile operating speed as measured by a Traffic and Engineering Survey exceeds the design speed , legal
protection is given to motorists traveling at such speeds ( design speed is `` based on conservative assumptions about driver , vehicle and roadway
characteristics '' ) .
</item 1>

<item 2>
Speed limit
In the European Union , large signposts showing the national ( default ) speed limits of the respective country are usually erected immediately after border
crossings , with a repeater sign some 200 to 500 m ( 660 to 1,640 ft ) after the first sign . Some places provide an additional `` speed zone ahead '' ahead of
the restriction and speed limit reminder signs may appear at regular intervals which may be painted on the road surface . Signs are normally placed on both
sides of the road and in some places there are small ( less than 1 / 4 the size of the sign ) rectangular orange reflector flags attached to both upper right
corners of both signs . The speed limit sign marking the new speed zone may also have the orange flags ; this practice can be observed in New York on highways
where the speed limit varies such as New York State Route 17 . Speed limits on European Motorways in km / h Signage in many countries , especially in Europe ,
conforms to the Vienna Convention on Road Signs and Signals using black text with a red circle on a white background . In the United States , the signs are
usually rectangular with the words `` SPEED LIMIT '' and the values in black on a white background .
</item 2>

<item 3>
Speed limit
The speeding fatality rate for local roads is three times that for Interstates Jump up ^ http://www.pe.com/articles/speed-776756-survey-limit.html Jump up ^
http://caselaw.findlaw.com/mt-supreme-court/1110919.html Jump up ^
http://vincenzeslaw.com/4-ways-virginia-reckless-driving-charges-handled-differently-fairfax-county-arlington-alexandria-stafford/ Jump up ^ `` Traffic
Operations : Road Signs '' . Government of South Australia . 2002 - 08 - 23 ( last update date ) . Retrieved 2010 - 09 - 26 . Most regulatory signs are
rectangular with a black legend on a white background ... ( exceptions include ) speed restriction signs ( symbol enclosed in a red circle ) '' Check date
values in : date = ( help ) Jump up ^ `` Speed Zoning Information '' ( PDF ) . Institute of Transportation Engineers . 22 March 2004 . Retrieved 2009 - 09 - 17
. Jump up ^ MUTCD Sections 2B.
</item 3>

<item 4>
Speed limit
The speed limit beyond the sign is the prevailing limit for the general area , for example the sign might be used to show the end of an urban area . In the
United Kingdom , the sign means that the national speed limit applies ( 60 mph on open roads , 70 mph on dual carriageways and motorways ) . In New Zealand it
means you are on an open road , but the maximum legal speed of 100 km / h still applies On roads without a general speed limits , such as German Autobahn , a
portion of Stuart Highway , and rural areas on the Isle of Man , it really means end of all quantitative speed limits . Common maximum speed limit derestriction
sign Common minimum speed limit derestriction sign Common advisory speed limit derestriction sign Australia Australia Belgium , United Kingdom , Republic of
Ireland pre-2005 , New Zealand , Singapore , Malaysia and Switzerland Japan In Japan , additional sign of `` ← '' or `` ここ まで '' is also derestriction sign .
But `` → '' and `` ここ から '' is `` from here '' .
</item 4>

<item 5>
Speed limit
Speed limit - wikipedia Speed limit Jump to : navigation , search This article is about road speed limits . For speed of light , see special relativity . For
road speed limits in specific countries , see Speed limits by country . For rail speed limits , see Slow zone . A typical 60 km / h speed limit sign used in
Australia A typical 50 mph speed limit sign in the United States A typical 50 mph speed limit sign in the United Kingdom and 50 km / h speed limit sign in Hong
Kong Road speed limits are used in most countries to set the maximum ( or minimum in some cases ) speed at which road vehicles may legally travel on particular
stretches of road . Speed limits may be variable and in some places speed is unlimited ( e.g. on some Autobahn sections in Germany ) . Speed limits are normally
indicated on a traffic sign . Speed limits are commonly set by the legislative bodies of nations or provincial governments and enforced by national or regional
police or judicial bodies . The first maximum speed limit was the 10 mph ( 16 km / h ) limit introduced in the United Kingdom in 1861 .
</item 5>

<item 6>
Speed limit
Per distance traveled , consequences of inappropriate speed are more frequent on lower speed , lower quality roads ; in the United States , for example , the ``
speeding fatality rate for local roads is three times that for Interstates '' Citations for violations of the basic speed law without a crash have sometimes
been ruled unfairly vague or arbitrary , hence a violation the due process of law , at least in the State of Montana . Even within states , differing
jurisdictions ( counties and cities ) choose to prosecute similar cases with differing approaches . Maximum speed limits ( edit ) A speed limit sign in Ontario
In France city sign has value of speed limit A street limited at 30 km / h End of speed limit , previous speed limit applies Most public roads in most countries
have a legally assigned numerical maximum speed limit which applies on all roads unless otherwise stated ; lower speed limits are often shown on a sign at the
start of the restricted section , although the presence of streetlights or the physical arrangement of the road may sometimes also be used instead . A posted
speed limit may only apply to that road or to all roads beyond the sign that defines them depending on local laws .
</item 6>

<item 7>
Speed limit
; for example California Vehicle Code section 22350 which states that `` No person shall drive a vehicle upon a highway at a speed greater than is reasonable
... and in no event at a speed which endangers the safety of persons or property '' . The basic speed law is almost always supplemented by specific maximum or
minimum limits , but applies regardless . The reasonable speed may be different than the posted speed limit in conditions such as fog , heavy rain , ice , snow
, gravel , sharp corners , blinding glare , darkness , crossing traffic , or when there is an obstructed view of orthogonal traffic -- such as by road curvature
, parked cars , vegetation , or snow banks -- thus limiting the Assured Clear Distance Ahead ( ACDA ) . Basic speed laws are statutized reinforcements of the
centuries - old common law negligence doctrine as specifically applied to vehicular speed .
</item 7>

<item 8>
Speed limit
In international European roads , speed should be taken into account at design stage . Maximum speed recommended by international European road design standard
Road classification 60 km / h 80 km / h ( 50 mph ) 100 km / h ( 60 mph ) 120 km / h ( 75 mph ) 140 km / h ( 85 mph ) Motorway x 80 100 120 140 Express road 60
80 100 120 x Road 60 80 100 x x Minimum speed limits ( edit ) UK minimum speed limit sign Some roads also have `` minimum speed limits '' , where slow speeds
can impede traffic flow or be dangerous . Signs often use blue circles based on the obligatory signs of the Vienna Convention on Road Signs and Signals . A
Japanese minimum speed sign has the same design as a maximum speed sign , but with a horizontal line below the number . In the United States of America , they
are also identical to their respective maximum speed limit signs with the exception of the text `` MINIMUM SPEED '' . This minimum speed is not so common , as
the risk usually raise with higher speed and not with lower speed .
</item 8>

<item 9>
Speed limit
Variable speed limits ( edit ) Example variable speed limit sign in the United States Digital speed limit sign for variable speed limits In 1965 , the first
known experiments with variable speed limit signs took place on a 30 km stretch of German motorway A8 between Munich and the border city of Salzburg , Austria .
Mechanically variable message signs could display speeds of 60 , 80 and 100 km / h , `` danger zone '' or `` accident '' . Personnel monitored traffic using
video technology , and manually controlled the signage . Beginning in the 1970s , additional advanced traffic control systems were put into service . Modern
motorway control systems can work without human intervention using various types of sensors to measure traffic flow and weather conditions . In 2009 , 1,300 km
( 810 mi ) of German motorways were equipped with such systems . In the late 1960s , heavily traveled portions of the New Jersey Turnpike began using variable
speed limit signs , in combination with variable message signs . Officials can adjust the speed limit according to weather , traffic conditions , and
construction .
</item 9>

<item 10>
Speed limit
p. 343 . External links ( edit ) Wikimedia Commons has media related to Speed limit signs and Driver speed and relative risk of an accident . Global map of
speed limits from Wikispeedia A Walk Through The History of Speed Limit Signs in The US John F. Carr 's State Traffic and Speed Laws Global map of speed limits
from OpenStreetMap Traffic signs Signs Warning sign Regulatory sign Prohibitory traffic sign Mandatory sign Special regulation sign Stop sign Yield sign
Crossbuck Dead End sign One - way traffic Speed limit ( by country ) Advisory speed limit Information sign Direction , position , or indication sign Variable -
message sign Bilingual sign Driver location signs Logo sign Priority to the right By comparison Comparison of European road signs Comparison of MUTCD -
influenced traffic signs Comparison of traffic signs in English - speaking countries By country Africa Botswana Mauritius South Africa Zimbabwe Asia Azerbaijan
Bangladesh Brunei Cambodia China Hong Kong India Indonesia Iran Israel Japan Laos Malaysia Nepal Pakistan Singapore South Korea Sri Lanka Thailand Turkey United
Arab Emirates Vietnam Europe Austria Bosnia and Herzegovina Czech Republic Denmark Estonia Finland France Germany Greece Iceland Ireland Italy Liechtenstein
Netherlands Norway Poland Sweden Switzerland Ukraine United Kingdom Scotland Wales North America Canada Mexico United States Oceania Australia New Zealand South
America Argentina Brazil Chile Colombia Lights Traffic
</item 10>
=== END ITEM id=b6e3c760019d123b ===

=== ITEM id=4fe8ef3e730af73d ===
Gold sentences and facts:
Sentence 0: The nineteenth season of Law & Order : Special Victims Unit premiered on September 27 , 2017 .
  Fact 0.0: The nineteenth season of Law & Order: Special Victims Unit premiered.
  Fact 0.1: The nineteenth season of Law & Order: Special Victims Unit premiered on September 27, 2017.

Retrieved items:
<item 1>
Law & Order: Special Victims Unit
Law & Order : Special Victims Unit - Wikipedia Law & Order : Special Victims Unit Jump to : navigation , search For the current season , see Law & Order :
Special Victims Unit ( season 19 ) . Law & Order : Special Victims Unit Also known as Law & Order : SVU SVU Genre Police procedural Legal drama Thriller Created
by Dick Wolf Starring Christopher Meloni Mariska Hargitay Richard Belzer Dann Florek Michelle Hurd Stephanie March Ice - T B.D. Wong Diane Neal Tamara Tunie
Adam Beach Michaela McManus Danny Pino Kelli Giddish Raúl Esparza Peter Scanavino Opening theme Theme of Law & Order : Special Victims Unit Composer ( s ) Mike
Post Country of origin United States Original language ( s ) English No. of seasons 19 No. of episodes 421 ( list of episodes ) Production Executive producer (
s ) Dick Wolf Robert Palm ( season 1 ) David J.
</item 1>

<item 2>
Law & Order: Special Victims Unit
ET , where it stayed until the twelfth season . In the 12th season , SVU moved back to 9 : 00 p.m. to lead in the newest Law & Order spinoff , Law & Order : LA
, until it was pulled from the network in January 2011 to be retooled . SVU moved back to 10 : 00 p.m. on January 12 , 2011 , until the end of the 13th season .
With season 14 , SVU moved back to 9 : 00 p.m. after a two - hour season premiere event on September 26 , 2012 , a position the show has held since . Russian
adaptation ( edit ) In 2007 , the Russian production company Studio 2B purchased the rights to create an adaptation of Law & Order : Special Victims Unit for
Russian television . Titled Закон и порядок : отдел оперативных расследований ( Law & Order : Division of Field Investigation ) , the series stars Alisa Bogart
as Major Olga Bobrova .
</item 2>

<item 3>
Law & Order: Special Victims Unit
2017 ( 2017 - 09 - 27 ) TBA TBA TBA Reception ( edit ) U.S. television Ratings ( edit ) Law & Order : Special Victims Unit premiered on Monday , September 20 ,
1999 . After nine episodes , the show was moved to Friday nights , where it found its audience and following its first and second seasons became a Top 20 -
overall show , overcoming the `` Friday night death slot '' phenomenon . Beginning with the season five , the show aired on Tuesdays to compete with CBS '
Judging Amy and ABC 's NYPD Blue . In its later years , Law & Order : Special Victims Unit consistently outperformed Law & Order in the Nielsen ratings for
first run episodes until the latter 's cancellation in 2010 . In 2016 , a New York Times study of the 50 TV shows with the most Facebook Likes found that SVU 's
popularity was `` atypical : generally slightly more popular in rural areas and the Black Belt , but largely restricted to the eastern half of the country . It
's most popular in Albany , N.Y. ; least in Colorado and Utah '' .
</item 3>

<item 4>
Law & Order: Special Victims Unit
1999 ( 1999 - 09 - 20 ) -- present Chronology Related shows Law & Order franchise External links Website Law & Order : Special Victims Unit ( often abbreviated
to Law & Order : SVU or just SVU ) is an American police procedural , legal , crime drama television series set in New York City , where it is also primarily
produced . In the style of the original Law & Order , episodes are often `` ripped from the headlines '' or loosely based on real crimes that have received
media attention . Created by Dick Wolf , the series premiered on NBC on September 20 , 1999 , as the second series in Wolf 's successful Law & Order franchise .
Law & Order : Special Victims Unit has been nominated for and won numerous awards , including the 2006 Emmy for Outstanding Lead Actress in a Drama Series for
Mariska Hargitay , the first , and to date only , Emmy to be received by a regular on any Law & Order series . It is the current longest running scripted
non-animated U.S.
</item 4>

<item 5>
Law & Order: Special Victims Unit (season 19)
Missing or empty title = ( help ) ( hide ) Law & Order : Special Victims Unit Characters Detectives Elliot Stabler Olivia Benson John Munch Monique Jeffries Fin
Tutuola Chester Lake Nick Amaro Amanda Rollins Dominick Carisi Jr . Commanding officers Donald Cragen Olivia Benson Assistant District Attorneys Alexandra Cabot
Casey Novak Kim Greylek Rafael Barba Doctors George Huang Melinda Warner Other characters Barry Moredock Brian Cassidy Elizabeth Donnelly Michael Cutter Sonya
Paxton Jack McCoy Seasons Episodes Season 1 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 Notable episodes `` Payback '' `` 911 '' `` Authority '' `` Cold '' ``
Zebras '' `` Behave '' `` Smoked '' `` Scorched Earth '' `` Justice Denied '' `` Father Dearest '' `` Rhodium Nights '' `` Lost Reputation '' `` Above Suspicion
'' `` Acceptable Loss '' `` Manhattan Vigil '' `` Comic Perversion '' `` Glasgowman 's Wrath '' `` Chicago Crossover '' See also Cast members Awards and
nominations Law & Order franchise Retrieved from `` https://en.
</item 5>

<item 6>
Law & Order: Special Victims Unit
Gillian Hardwicke served as the SVU ADA during season twelve . In season thirteen , both Cabot and Novak returned as ADAs and since the beginning of season
fourteen , ADA Rafael Barba has been SVU 's prosecutor . In season 13 other big changes happened when Stabler left in the season twelve finale . Detectives Nick
Amaro and Amanda Rollins joined the team filling the void left by Stabler . Amaro brings empathy to his cases while dealing with a stressful home life , whilst
Rollins ' dogged persistence and instincts help her close cases , but her secrets could derail her career . In the current seasons the Manhattan SVU is now run
by Lieutenant Olivia Benson . Episodes ( edit ) Main article : List of Law & Order : Special Victims Unit episodes Law & Order : Special Victims Unit has
concluded its eighteenth season and has started its nineteenth season . Each season has aired on NBC and consists of 19 to 25 episodes , each lasting
approximately forty minutes ( sixty minutes including commercials ) . Season Episodes Originally aired Nielsen ratings First aired Last aired Rank Rating 22
September 20 ,
</item 6>

<item 7>
Law & Order: Special Victims Unit
ET for the first nine episodes , from September 20 through November 29 , 1999 . It was then shifted to Friday nights at 10 p.m. ET on January 7 , 2000 , and
remained in that time slot through the end of season four on May 16 , 2003 . Beginning with the season five premiere on September 23 , 2003 , SVU moved to
Tuesday nights at 10 : 00 p.m. ET . NBC airs repeats on Saturday nights at varying times and previous episodes are shown on the USA Network on varying days in
marathon blocks With the season eleven premiere on September 23 , 2009 , the series vacated its Tuesday 10 p.m. ET slot because NBC began a prime - time
weeknight Jay Leno series . The new time slot became Wednesday nights at 9 : 00 p.m. ET on NBC , with CTV still airing SVU on Tuesdays at 10 : 00 in Canada .
After the 2010 Winter Olympics on March 3 , 2010 , the time slot for SVU changed again to Wednesdays at 10 p.m.
</item 7>

<item 8>
Law & Order: Special Victims Unit
The first episode , `` Payback '' , premiered on NBC on September 20 , 1999 . Executive producers ( edit ) Executive producer Neal Baer left Law & Order : SVU
as showrunner at the end of season twelve , after eleven years ( seasons 2 -- 12 ) on the show , in order to sign a three - year deal with CBS Studios . Baer
was replaced by former Law & Order : Criminal Intent showrunner Warren Leight . In March 2015 , it was announced that Warren Leight signed a three - year deal
with Sony Pictures Television , that will allow him to work on SVU one more season , its seventeenth . Leight joined the show in season thirteen . It was
announced on March 10 , 2016 that original Law & Order veteran producer Rick Eid would take Leight 's place as showrunner starting in season 18 . Creator Dick
Wolf commented to The Hollywood Reporter , `` I 'm extremely pleased that Rick had decided to rejoin the family and hope that he will be here for years to come
. ''
</item 8>

<item 9>
Law & Order: Special Victims Unit
During post-production of season 18 , following the announcement that SVU was renewed for a nineteenth season , it was revealed that Rick Eid departed the
series . He will be taking over another Dick Wolf / NBC series . Chicago P.D. It was announced on May 25 , 2017 that original Law & Order and Law & Order :
Criminal Intent showrunner Michael S. Chernuchin would be reprising his role starting on season nineteen . Chernuchin was also co-creator and executive
producing showrunner of Chicago Justice , another Wolf - related show that was canceled by NBC at the end of the 2016 -- 17 TV season . Filming ( edit ) SVU
shooting on location in Central Park at night Many exterior scenes of Law & Order : Special Victims Unit are filmed on location in New York City , Wolf 's
hometown , throughout all five of New York City 's boroughs . As the NYPD encounters varied law enforcement challenges on a daily basis , the backdrop provides
the writers a supply of ideal locations from which to choose . Fort Lee , New Jersey served as the filming location for Detective Elliot Stabler 's residence in
Queens , New York .
</item 9>

<item 10>
Law & Order: Special Victims Unit
The show also briefly ran on Syfy in 2006 . International broadcast ( edit ) In Australia , Law & Order : SVU airs on Network Ten and Australia 's cable channel
Universal Channel ( formerly aired on TV1 which ceased to broadcast in December 2013 ) . In the UK `` Law & Order : SVU '' originally aired on the Hallmark
Channel ( which in 2012 became the Universal Channel ) with a terrestrial TV airing on Channel 5 in 2016 which then switched to Five USA . The UK is two and
half weeks behind the US . In Brazil , Law & Order : SVU also airs on Universal Channel , with weekdays episodes from past seasons and simultaneously , two
episodes after the original airing in the US . Law & Order : SVU is broadcast at 9.30 pm on Tuesdays on TV3 in New Zealand . Series 18 premiered on 11 April
2017 . References ( edit ) Notes ( edit ) Jump up ^ SVU Writers Room ( @ SVUWritersRoom ) ( 7 July 2016 ) . `` First cast read - through of # SVUSeason18 '' (
Tweet ) -- via Twitter .
</item 10>
=== END ITEM id=4fe8ef3e730af73d ===

=== ITEM id=c256bf7ca1ec7b63 ===
Gold sentences and facts:
Sentence 0: John Ernest Crawford ( born March 26 , 1946 ) is an American character actor , singer , and musician . At age 12 , Crawford rose to fame for playing Mark McCain , the son of Lucas McCain ( played by Chuck Connors ) , in the popular ABC Western series , The Rifleman , which originally aired from 1958 to 1963 . Crawford first performed before a national audience as a Mouseketeer .
  Fact 0.0: John Ernest Crawford was born March 26, 1946.
  Fact 0.1: John Ernest Crawford is an American character actor.
  Fact 0.2: John Ernest Crawford is an American singer.
  Fact 0.3: John Ernest Crawford is an American musician.
  Fact 0.4: At age 12, John Ernest Crawford rose to fame for playing Mark McCain.
  Fact 0.5: Mark McCain is the son of Lucas McCain.
  Fact 0.6: Lucas McCain was played by Chuck Connors.
  Fact 0.7: Crawford played Mark McCain in the popular ABC Western series The Rifleman.
  Fact 0.8: The Rifleman originally aired from 1958 to 1963.
  Fact 0.9: John Ernest Crawford first performed before a national audience as a Mouseketeer.

Retrieved items:
<item 1>
Rio Bravo (film)
Because the film starred a crooner , Martin , and a teen idol , Nelson , Hawks included three songs in the soundtrack . Before the big showdown , in the jail
house , Martin sings `` My Rifle , My Pony , and Me '' ( which contained new lyrics to a Tiomkin tune that appeared in Red River ) accompanied by Nelson , after
which Nelson sings a brief version of `` Get Along Home , Cindy '' , accompanied by Martin and Brennan . Over the closing credits , Martin , backed by the
Nelson Riddle Orchestra , sings a specially composed song , `` Rio Bravo '' , written by Tiomkin with lyrics by Paul Francis Webster . Nelson later paid homage
to both the film and his character , Colorado , by including the song `` Restless Kid '' on his 1959 LP , Ricky Sings Again . Members of the Western Writers of
America chose `` My Rifle , My Pony , and Me '' as one of the Top 100 Western songs of all time . A brief clip from Rio Bravo was among the archive footage
later incorporated into the opening sequence of Wayne 's last film , The Shootist , to illustrate the backstory of Wayne 's character .
</item 1>

<item 2>
Rio Bravo (film)
raˈtʃon ) , Spanish for `` big drunk '' ) , enters a saloon but can not afford a drink . Joe Burdette ( Claude Akins ) , brother of rancher Nathan Burdette ,
watches as a barfly ( Myron Healey ) , tosses a silver dollar into a spittoon . Presidio County , Texas , Sheriff John T. Chance ( John Wayne ) appears and
kicks the spittoon away , looking at Dude with disgust . Dude is shamed by his plight and takes out his anger on Chance , knocking him out with an ax handle .
Joe begins punching Dude , then shoots and kills an unarmed bystander who tries to intervene . Joe heads to his brother 's saloon , where a bloody Chance comes
to arrest him for the murder of the bystander . Another patron draws his gun on Chance , but Dude shoots the gun out of the man 's hand . Joe is locked up in
the local jail . Chance is willing to deputize Dude , provided he can stay sober . Both remember how good with a gun Dude used to be .
</item 2>

<item 3>
Rio Bravo (film)
Both remember how good with a gun Dude used to be . Chance 's friend Pat Wheeler ( Ward Bond ) and his wagon train of supplies stop in town , with a young
gunslinger , Colorado Ryan ( Ricky Nelson ) , riding guard . Inside the jail , Stumpy ( Walter Brennan ) , Chance 's game - legged deputy , keeps watch over the
jail and Joe , who knows that Stumpy holds an old grudge against Joe 's wealthy and powerful brother . Joe warns his jailers that Nathan Burdette will not like
how his brother is being treated . A mysterious woman nicknamed Feathers ( Angie Dickinson ) is in the saloon , playing poker . In the meantime , Dude and
Chance patrol the town . Hotel owner Carlos ( Pedro Gonzalez - Gonzalez ) stops the sheriff , saying Wheeler has been talking too much about Chance needing help
. In the saloon , Chance implores Wheeler to stop , as it will draw attention from the wrong people . Wheeler suggests that Colorado could be of assistance ,
but Colorado politely declines , saying he wants to `` mind my own business '' . Colorado also promises not to start any trouble without telling the sheriff
first .
</item 3>

<item 4>
Rio Bravo (film)
In this film , Robert Mitchum played the Dean Martin role , Arthur Hunnicutt the Walter Brennan character , and James Caan the Ricky Nelson role . Hawks again
named the Nelson / Caan character after a state ( in this case , Mississippi ) and in a wry , humorous twist on the original film , Hawks made him inept with
firearms , but skilled with a knife . The second remake , Rio Lobo , was made in 1970 with a plot much further off the original mold , starting with the absence
of a lawman - turned - drunkard character . This began with a Confederate train robbery of a Union gold shipment during the American Civil War , then moved to a
postwar Texas county thoroughly controlled by a rich , arrogant rancher . The heroes , with the exception of an old man similar to Brennan and Hunnicutt 's
characters in the previous pictures ( Jack Elam here ) , were complete outsiders . Along with Wayne and Elam , this movie starred Mexican film star Jorge Rivero
( as Frenchie ) , Christopher Mitchum ( Robert Mitchum 's son ) , and Jennifer O'Neill .
</item 4>

<item 5>
Johnny Crawford
His first important break as an actor followed with the title role in a Lux Video Theatre production of `` Little Boy Lost '' , a live NBC broadcast on March 15
, 1956 . He also appeared in the popular Western series The Lone Ranger , in 1956 , in one of the few color episodes of that series . Following that performance
, the young actor worked steadily with many seasoned actors and directors . Freelancing for 2 1 / 2 years , he accumulated almost 60 television credits ,
including featured roles in three episodes of NBC 's The Loretta Young Show and an appearance as Manuel in , `` I Am an American '' , an episode of the
syndicated crime drama Sheriff of Cochise . By the spring of 1958 , he had also performed 14 demanding roles in live teleplays for NBC 's Matinee Theatre ,
appeared on CBS 's sitcom , Mr. Adams and Eve , in the Wagon Train episode `` The Sally Potter Story '' ( in which Martin Milner also appeared ) and on the
syndicated series , Crossroads , Sheriff of Cochise , and Whirlybirds , and made three pilots of TV series .
</item 5>

<item 6>
Rio Bravo (film)
After a pair of sheer tights comes floating from a window to the street , a delighted Stumpy retrieves it , but Dude cautions him to mind his own business .
Cast ( edit ) John Wayne and Angie Dickinson in Rio Bravo John Wayne as John T. Chance Dean Martin as Dude Ricky Nelson as Colorado / Ryan Angie Dickinson as
Feathers Walter Brennan as Stumpy Ward Bond as Pat Wheeler John Russell as Nathan Burdette Pedro Gonzalez Gonzalez as Carlos Robante Estelita Rodriguez as
Consuela Robante Claude Akins as Joe Burdette Joseph Shimada as Burt Malcolm Atterbury and Harry Carey Jr. also receive a screen credit in the film 's opening ,
but their scenes were deleted from the final film . Production ( edit ) Ricky Nelson performing the song `` Get Along Home , Cindy '' in Rio Bravo Exteriors for
the film were shot at Old Tucson Studios , just outside Tucson . Filming took place in the summer of 1958 , and the movie 's credits gave 1958 as the year of
production , although the film was not released until 1959 .
</item 6>

<item 7>
Johnny Crawford
He played a young deputy Billy Norris , in The Big Valley episode `` The Other Face Of Justice '' in 1969 . While enlisted in the United States Army for two
years , Crawford worked on training films as a production coordinator , assistant director , script supervisor , and occasional actor . His rank was sergeant at
the time of his honorable discharge in December 1967 . In 1968 , Crawford played a soldier wanted for murder in `` By the Numbers '' , an episode of the popular
TV series Hawaii Five - O . The Resurrection of Broncho Billy was a USC student film Crawford agreed to do as a favor to his close friend , producer John
Longenecker . It won the 1970 Academy Award for Best Live Action Short Subject . The Naked Ape was a partially animated 1973 feature film starring Crawford and
Victoria Principal , produced by Hugh Hefner . In 1976 , Johnny Crawford co-starred as Ben Shelby in the tenth episode of season three of Little House on the
Prairie , `` The Hunters '' . In 1985 , Crawford appeared as Noah Paisley , sheriff 's deputy , in an episode of Murder , She Wrote .
</item 7>

<item 8>
Johnny Crawford
The third pilot , which was made as an episode of Dick Powell 's Zane Grey Theater , was picked up by ABC and the first season of The Rifleman began filming in
July 1958 . Crawford was nominated for an Emmy Award , at age 13 , for his role as Mark McCain , the son of Lucas McCain , played by Chuck Connors , in the Four
Star Television series The Rifleman , which originally aired from 1958 to 1963 . Throughout The Rifleman 's five seasons , a remarkable on - screen chemistry
existed between Connors and Crawford in the depiction of their father - son relationship . They were still close friends when Connors died on November 10 , 1992
, and Crawford gave a eulogy at Connors ' memorial service . During the late 1950s and early 1960s , Crawford had wide popularity with American teenagers and a
recording career that generated four Billboard Top 40 hits , including the single , `` Cindy 's Birthday '' , which peaked at number 8 , in 1962 .
</item 8>

<item 9>
Johnny Crawford
Jump up ^ `` Johnny Crawford '' . Scott Stander . Retrieved January 25 , 2014 . Jump up ^ `` Angry Woman -- Victoria Jackson on `` Tonight Show '' b / w `` .
Youtube . Retrieved January 25 , 2014 . Biography portal California portal Los Angeles portal Film portal Television portal Music portal United States Army
portal External links ( edit ) Johnny Crawford on The Rifleman Broncho Billy short film Johnny Crawford 's MySpace site Johnny Crawford on IMDb VIAF : 63364407
LCCN : no91003539 ISNI : 0000 0001 1526 4402 GND : 142142425 SUDOC : 079205798 BNF : cb11219552n ( data ) MusicBrainz : bf40b9eb - 068a - 4bdd - be46 -
ec0b5bd7a177 BNE : XX4832002 SNAC : w6sk5xbc Retrieved from `` https://en.wikipedia.org/w/index.
</item 9>

<item 10>
Johnny Crawford
Johnny Crawford - wikipedia Johnny Crawford Jump to : navigation , search For the Canadian ice hockey player and coach , see Johnny Crawford ( ice hockey ) .
Johnny Crawford Johnny Crawford in The Rifleman ( 1961 ) John Ernest Crawford ( 1946 - 03 - 26 ) March 26 , 1946 ( age 72 ) Los Angeles , California , U.S.
Occupation Actor , singer , musician , band leader Years active 1954 -- 96 Spouse ( s ) Charlotte Crawford ( 1995 -- present ) ( separated ) Relatives Robert L.
Crawford , Jr. ( brother ) Website Johnny Crawford fansite Johnny Crawford and Chuck Connors in The Rifleman ( 1960 ) John Ernest Crawford ( born March 26 ,
1946 ) is an American character actor , singer , and musician . At age 12 , Crawford rose to fame for playing Mark McCain , the son of Lucas McCain ( played by
Chuck Connors ) , in the popular ABC Western series , The Rifleman , which originally aired from 1958 to 1963 . Crawford first performed before a national
audience as a Mouseketeer .
</item 10>
=== END ITEM id=c256bf7ca1ec7b63 ===

=== ITEM id=c9543437f8b0ccd6 ===
Gold sentences and facts:
Sentence 0: The Country Music Hall of Fame and Museum in Nashville , Tennessee , is one of the world 's largest museums and research centers dedicated to the preservation and interpretation of American vernacular music . Chartered in 1964 , the museum has amassed one of the world 's most extensive musical collections .
  Fact 0.0: The Country Music Hall of Fame and Museum is in Nashville, Tennessee.
  Fact 0.1: The Country Music Hall of Fame and Museum is one of the world's largest museums dedicated to the preservation and interpretation of American vernacular music.
  Fact 0.2: The Country Music Hall of Fame and Museum is one of the world's largest research centers dedicated to the preservation and interpretation of American vernacular music.
  Fact 0.3: The Country Music Hall of Fame and Museum was chartered in 1964.
  Fact 0.4: The Country Music Hall of Fame and Museum has amassed one of the world's most extensive musical collections.
Sentence 1: Country Music Hall of Fame and Museum Established 1961 ( 1961 ) Location 222 5th Ave S Nashville , Tennessee 37203 Type Hall of fame Visitors 1.1 million in 2015 Director Kyle Young Website www.countrymusichalloffame.org
  Fact 1.0: The Country Music Hall of Fame and Museum was established in 1961.
  Fact 1.1: The Country Music Hall of Fame and Museum is located at 222 5th Ave S, Nashville, Tennessee 37203.
  Fact 1.2: The type of the Country Music Hall of Fame and Museum is Hall of fame.
  Fact 1.3: The Country Music Hall of Fame and Museum had 1.1 million visitors in 2015.
  Fact 1.4: The director of the Country Music Hall of Fame and Museum is Kyle Young.
  Fact 1.5: The website of the Country Music Hall of Fame and Museum is www.countrymusichalloffame.org.

Retrieved items:
<item 1>
Country Music Hall of Fame and Museum
Award recipients are determined through a two - stage balloting process ; the first round of voting narrows each category to five candidates ; the second round
selects winners . The large select committee of electors that votes on Hall of Fame membership is composed of CMA members who have participated in the country
music industry for at least ten years . New Hall of Fame members receive special recognition in ceremonies at the Country Music Hall of Fame and Museum . Bas -
relief portraits cast in bronze honoring each Hall of Fame member were originally displayed at the Tennessee State Museum in downtown Nashville until the
Country Music Hall of Fame and Museum opened its own building in April 1967 ; in this barn - roofed facility at the head of Music Row , the bronze plaques
comprised a special exhibit . Today the plaques are displayed in a seventy - foot - high rotunda at the museum 's enlarged downtown Nashville facility . The
museum collection ( edit ) The museum 's archival and library storage , allows for the cultivation of the museum 's collection and includes more than 2.5
million artifacts .
</item 1>

<item 2>
Country Music Hall of Fame and Museum
Located at the head of Music Row , the museum was erected on the site of a small Nashville city park . At this point , artifacts began to be displayed and a
small library was begun in a loft above one of the museum 's galleries . Early in the 1970s the basement of the museum building was partially complete , and
library expansion began , embracing not only recordings but also books and periodicals , sheet music and songbooks , photographs , business documents , and
other materials . At the outset , CMA staff had run the museum , but by 1972 the museum ( already governed by its own independent board of directors ) acquired
its own small staff , which has steadily increased to over 150 full - time professionals . Building expansion took place in 1974 , 1977 , and 1984 to store and
display the museum 's growing collection of costumes , films , historic cars , musical instruments , and other artifacts . An education department was created
to conduct ongoing programs with Middle Tennessee schools , an oral history program was begun , and a publications department was launched to handle books , as
well as the Journal of Country Music .
</item 2>

<item 3>
Country Music Hall of Fame and Museum
Current museum ( edit ) To become more accessible , in May 2001 the Country Music Hall of Fame and Museum moved to a new , 130,000 - square - foot ( 12,000 m )
facility in the heart of downtown Nashville 's arts and entertainment district . In 2014 , the museum unveiled a $100 million expansion , doubling its size to
350,000 square feet of galleries , archival storage , education classrooms , retail stores , and special event space . In the museum 's core exhibition , Sing
Me Back Home : A Journey Through Country Music , visitors are immersed in the history and sounds of country music , its origins and traditions , and the stories
and voices of many of its architects . The story is revealed through artifacts , photographs , and text panels , recorded sound , vintage video , and
interactive touchscreens . Sing Me Back Home is enhanced by numerous , rotating limited - engagement exhibits . The new ACM Gallery and the Dinah and Fred
Gretsch Family Gallery offer visitors a hands - on immersion into today 's country music with artifacts from today 's country stars and a series of technology -
enhanced activities .
</item 3>

<item 4>
Country Music Hall of Fame and Museum
Country Music Hall of Fame and museum - wikipedia Country Music Hall of Fame and museum Jump to : navigation , search This article needs additional citations
for verification . Please help improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( May 2016 ) (
Learn how and when to remove this template message ) Country Music Hall of Fame and Museum Established 1961 ( 1961 ) Location 222 5th Ave S Nashville ,
Tennessee 37203 Type Hall of fame Visitors 1.1 million in 2015 Director Kyle Young Website www.countrymusichalloffame.org The Country Music Hall of Fame and
Museum in Nashville , Tennessee , is one of the world 's largest museums and research centers dedicated to the preservation and interpretation of American
vernacular music . Chartered in 1964 , the museum has amassed one of the world 's most extensive musical collections .
</item 4>

<item 5>
Country Music Hall of Fame and Museum
Nashville , Tenn. : Country Music Foundation , 1990 . N.B. : The structure as described in this document is the original one which had opened in 1967 ; the book
( amply ill . in b&w and col . ) describes the work , collections , etc. of the center 's Hall of Fame , Museum ( which includes a library ) , and recording
studio , and also gives an account of the development and various aspects of the country music industry . ISBN 0 - 915608 - 12 - X References ( edit ) ^ Jump up
to : `` About '' . Country Music Hall of Fame . Retrieved January 14 , 2016 . Jump up ^ McCall , Michael ( 2012 ) . The Encyclopedia of Country Music Second
Edition . Oxford University Press . pp. 105 -- 106 . ISBN 0195395638 . Jump up ^ `` Home '' . Studio B. Retrieved August 26 , 2017 . Jump up ^ `` Country Music
Hall of Fame promotes four to VP roles '' . Nashville Post .
</item 5>

<item 6>
Country Music Hall of Fame and Museum
Contents ( hide ) 1 History of the museum 2 Current museum 3 The Country Music Hall of Fame 4 The Museum collection 5 Architecture and design 6 Timeline 7
Senior leadership 8 See also 9 Further reading 10 References 11 External links History of the museum ( edit ) The Country Music Hall of Fame and Museum is one
of the world 's largest and most active popular music research centers and the world 's largest repository of country music artifacts . Early in the 1960s , as
the Country Music Association 's campaign to publicize country music was accelerating , CMA leaders determined that a new organization was needed to operate a
country music museum and to carry out research and education activities beyond CMA 's scope as a trade organization . Toward this end , the nonprofit Country
Music Foundation ( CMF ) was chartered by the state of Tennessee in 1964 to collect , preserve , and publicize information and artifacts relating to the history
of country music . Through CMF , industry leaders raised money to build the Country Music Hall of Fame and Museum , which opened on April 1 , 1967 .
</item 6>

<item 7>
Country Music Hall of Fame and Museum
Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use
and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 7>

<item 8>
Country Music Hall of Fame and Museum
In addition to the galleries , the museum has the 776 - seat CMA Theater , the Taylor Swift Education Center , and multi-purpose event rental spaces . Other
historic properties of the Country Music Hall of Fame and Museum include the letterpress operation Hatch Show Print ( located inside the museum ) and Historic
RCA Studio B ( located on Music Row ) , Nashville 's oldest surviving recording studio , where recordings by Elvis Presley , Dolly Parton , Waylon Jennings ,
and many others were made . The Country Music Hall of Fame and Museum has developed multiple platforms to make its collection accessible to a wider audience .
From weekly instrument demonstrations to its flagship songwriting program for schools , Words & Music , the museum offers an aggressive schedule of educational
programs . The museum also operates CMF Records , a Grammy - winning re-issue label ( The Complete Hank Williams and Night Train to Nashville : Music City
Rhythm & Blues , 1945 - 1970 ) ; and CMF Press , a book publishing arm that releases exhibit - related books in cooperation with Vanderbilt University Press and
other major trade publishing houses .
</item 8>

<item 9>
Graph query row:
organization: Country Music Hall of Fame and Museum
city: Nashville
state: Tennessee
</item 9>

<item 10>
Country Music Hall of Fame and Museum
Nashville Post . February 17 , 2015 . Retrieved January 19 , 2016 . External links ( edit ) Wikimedia Commons has media related to Country Music Hall of Fame
and Museum . Official website ( hide ) Nashville landmarks Existing AT&T Building Belmont Mansion Bicentennial Capitol Mall State Park Bluebird Café Bridgestone
Arena Centennial Park Cheekwood Country Music Hall of Fame First Tennessee Park Fort Nashborough Fort Negley Frist Center for the Visual Arts Gaylord Opryland
Resort Grand Ole Opry Greer Stadium The Hermitage Life & Casualty Tower Memorial Gymnasium Music City Center Nashville City Cemetery Nashville International
Airport Nashville Zoo at Grassmere Nissan Stadium Parthenon Ryman Auditorium Schermerhorn Symphony Center John Seigenthaler Pedestrian Bridge Tennessee
Performing Arts Center Tennessee State Capitol Tennessee State Museum Tennessee Tower Union Station Vanderbilt Stadium Former Opryland USA Starwood Amphitheatre
Sulphur Dell Coordinates : 36 ° 09 ′ 29.80 '' N 86 ° 46 ′ 34.18 '' W ﻿ / ﻿ 36.1582778 ° N 86.7761611 ° W ﻿ / 36.
</item 10>
=== END ITEM id=c9543437f8b0ccd6 ===
