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

This file has 7 items.

=== ITEM id=a80c1125e1852e19 ===
Gold sentences and facts:
Sentence 0: In 2012 , Maryse was featured on the cover of Le Journal de Montréal , and on MDA Show of Strength with The Miz and other celebrities . In 2015 , Ouellet had roles in Sharknado 3 : Oh Hell No ! and in Santa 's Little Helper , the latter starring Ouellet 's husband , The Miz . In the same year , she appeared on the E ! reality television series WAGS . In 2016 , it was announced that Maryse would appear as a main cast member on the sixth season of Total Divas , which prompted her return to WWE . In June 2017 , Maryse along with Miz were cover of the DUB Magazine with a Jeep car . In 2018 , Maryse along with The Miz will star in a reality television series titled Miz & Mrs .
  Fact 0.0: In 2012, Maryse was featured on the cover of Le Journal de Montréal.
  Fact 0.1: In 2012, Maryse was featured on MDA Show of Strength with The Miz.
  Fact 0.2: In 2012, Maryse was featured on MDA Show of Strength with other celebrities.
  Fact 0.3: In 2015, Ouellet had a role in Sharknado 3: Oh Hell No!
  Fact 0.4: In 2015, Ouellet had a role in Santa's Little Helper.
  Fact 0.5: Santa's Little Helper starred Ouellet's husband, The Miz.
  Fact 0.6: In 2015, Ouellet appeared on the E! reality television series WAGS.
  Fact 0.7: In 2016, it was announced that Maryse would appear as a main cast member on the sixth season of Total Divas.
  Fact 0.8: The 2016 announcement prompted Maryse's return to WWE.
  Fact 0.9: In June 2017, Maryse and Miz were on the cover of DUB Magazine with a Jeep car.
  Fact 0.10: In 2018, Maryse and The Miz will star in a reality television series titled Miz & Mrs.
Sentence 1: In late - 2011 , she announced plans for a clothing and jewelry line called House of Maryse , and later began working as a realtor . In November 2016 , she began starring in the reality television series Total Divas on E ! as part of the main cast . She and her husband The Miz will also star in their own reality show titled Miz & Mrs. that will premiere in 2018 .
  Fact 1.0: In late 2011, Maryse announced plans for a clothing and jewelry line called House of Maryse.
  Fact 1.1: Maryse later began working as a realtor.
  Fact 1.2: In November 2016, Maryse began starring in the reality television series Total Divas on E! as part of the main cast.
  Fact 1.3: Maryse and her husband The Miz will also star in their own reality show titled Miz & Mrs.
  Fact 1.4: The reality show Miz & Mrs. will premiere in 2018.

Retrieved items:
<item 1>
Maryse Ouellet
In late - 2011 , she announced plans for a clothing and jewelry line called House of Maryse , and later began working as a realtor . In November 2016 , she
began starring in the reality television series Total Divas on E ! as part of the main cast . She and her husband The Miz will also star in their own reality
show titled Miz & Mrs. that will premiere in 2018 . Contents ( hide ) 1 Early life and career 2 Professional wrestling career 2.1 World Wrestling Entertainment
/ WWE 2.1. 1 Diva Search and developmental territories ( 2006 -- 2007 ) 2.1. 2 Early feuds ( 2006 -- 2008 ) 2.1. 3 Divas Champion ( 2008 -- 2010 ) 2.1. 4
Managing Ted DiBiase and departure ( 2010 -- 2011 ) 2.2 Independent circuit ( 2012 ) 2.3 Return to WWE 2.3.
</item 1>

<item 2>
Maryse Ouellet
She began appearing regularly for FWE , where she commentated during women 's matches . Return to WWE ( edit ) Managing the Miz ( 2016 -- present ) ( edit )
Maryse celebrating along The Miz after winning the Intercontinental Championship at a Raw event in April 2016 On April 4 , 2016 episode of Raw , the episode
after WrestleMania 32 , Maryse returned during a WWE Intercontinental Championship match between the champion Zack Ryder and her husband , The Miz . During the
match , Maryse slapped Ryder 's father , who was in the front row , in order to distract Ryder and allowed Miz to capture the championship for the fifth time ,
thus becoming his manager in the process , where the two then became an on - screen power couple . She formally re-introduced herself three days later on the
April 7 episode of SmackDown before once more helping Miz to retain his championship over Ryder . She then began to cut various promos with Miz as his host
during the `` Miz TV '' segments , while calling themselves the `` It '' Couple .
</item 2>

<item 3>
Maryse Ouellet
In 2018 , Maryse along with The Miz will star in a reality television series titled Miz & Mrs . Maryse has appeared in four WWE video games . She made her in -
game debut at WWE Smackdown vs. Raw 2010 , later appearing in WWE Smackdown vs. Raw 2011 , WWE ' 12 , and returned in WWE 2K18 . Filmography ( edit ) Film (
edit ) Year Title Role Notes Octobre Pouilleuse Film debut 2015 Sharknado 3 : Oh Hell No !
</item 3>

<item 4>
Maryse Ouellet
On September 20 , Miz retained the championship over Ziggler . On October 9 , at No Mercy Miz lost the Intercontinental Championship to Ziggler in a title vs
career match when Maryse was banned from the ringside along with The Spirit Squad ( Kenny and Mikey ) , who was contracted by her to help Miz , ending Miz 's
reign at 188 days . On November 8 episode of SmackDown , Maryse went on to speak in Miz 's name due Miz refuse to speak with Daniel Bryan . At that night ,
Maryse accepted Bryan 's offer to Miz face Ziggler the following week for the Intercontinental Championship in the 900th episode of SmackDown . Later that night
, Bryan would state on Talking Smack that if Ziggler successfully retains his championship , The Miz and Maryse would be traded to Raw for someone else . On
November 15 at the 900th episode of SmackDown , Maryse helped Miz to win the championship for the sixth time and also retain it at Survivor Series against Sami
Zayn .
</item 4>

<item 5>
Maryse Ouellet
Following a hiatus due to the shoot of a WWE film , Maryse returned to WWE television on the June 27 episode of Raw , where she helped Miz to retain his
championship against Kane after she faked an ankle injury . After being drafted to Smackdown as part of the 2016 WWE draft which took place on July 19 , Ouellet
made her first appearance for the brand on July 26 , where she co-hosted a Miz TV segment along her husband before being interrupted by Randy Orton , who
challenged Miz to a non-title match , in which he was subsequently defeated . At Backlash , Maryse successfully helped Miz to retain the Intercontinental
Championship by spraying something at Dolph Ziggler whilst Miz distracted the referee . The following week , SmackDown General Manager Daniel Bryan announced a
rematch for the championship due Maryse 's interference at Backlash . On September 20 , Miz retained the championship over Ziggler .
</item 5>

<item 6>
Maryse Ouellet
She is currently inactive due to her maternity leave . After spending years modeling , including winning Miss Hawaiian Tropic Canada in 2003 , Ouellet was hired
by the professional wrestling promotion WWE in August 2006 , after she participated in the WWE Diva Search competition . She spent time in Ohio Valley Wrestling
and Florida Championship Wrestling , WWE 's developmental territories , before being assigned to the SmackDown brand in 2008 . In December 2008 , Maryse won her
first WWE Divas Championship and held it for seven months , which is the third longest reign in the championship 's history . In April 2009 , she was drafted to
the Raw brand , and won the WWE Divas Championship for the second time in February 2010 , making her the first wrestler to have held the championship on more
than one occasion . In mid-2010 , she became the co-host of NXT and the manager of Ted DiBiase , prior to her release from the company in October 2011 . In
April 2016 , she returned to the company and became the manager of her husband The Miz , managing him to three Intercontinental Championship reigns .
</item 6>

<item 7>
Maryse Ouellet
Maryse Ouellet - wikipedia Maryse Ouellet Jump to : navigation , search Maryse Ouellet Ouellet in April 2016 Birth name Maryse Ouellet ( 1983 - 01 - 21 )
January 21 , 1983 ( age 35 ) Montréal , Quebec , Canada Residence Austin , Texas , United States Spouse ( s ) The Miz ( m . 2014 ) Children Professional
wrestling career Ring name ( s ) Maryse Billed height 5 ft 8 in ( 1.73 m ) Billed weight 116 lb ( 53 kg ) Billed from Montreal , Quebec , Canada Trained by Al
Snow Dave Finlay Steve Keirn Tom Prichard Ricky Steamboat Debut August 24 , 2006 Maryse Mizanin ( / məˈriːs / ; née Ouellet ; born January 21 , 1983 ) is a
French Canadian professional wrestler , professional wrestling manager , actress , businesswoman and glamour model currently signed to WWE under the ring name
Maryse , performing on the SmackDown brand as the manager of her real - life husband The Miz .
</item 7>

<item 8>
Maryse Ouellet
. TMZ . February 22 , 2014 . Retrieved March 21 , 2014 . Jump up ^ `` The Miz and Maryse are expecting their first child '' . Retrieved March 19 , 2018 . Jump
up ^ Waldman , Jon ( April 25 , 2009 ) . `` Smackdown : UK gets a strong episode of SD ! '' . Slam ! Sports . Canadian Online Explorer . Retrieved February 14 ,
2010 . Jump up ^ David , Stephens ( January 4 , 2010 ) . `` WWE Raw Results -- 1 / 4 / 10 '' . WrestleView . Retrieved April 9 , 2016 . Jump up ^ David ,
Stephens ( May 17 , 2010 ) . `` WWE Raw Results -- 5 / 17 / 10 '' . WrestleView . Retrieved April 9 , 2016 . Jump up ^ Tedesco , Mike ( December 26 , 2008 ) .
</item 8>

<item 9>
Maryse Ouellet
On June 4 , at Extreme Rules , Maryse was in Miz 's corner when he won the WWE Intercontinental Championship for the seventh time by defeating Ambrose . In
September , following her pregnancy announcement , Maryse stopped appearing on television . On January 22 , 2018 , in the Raw 25 Years special episode , Maryse
was honored as part of a segment involving women considered legends that contributed to the company 's success , including the Bella Twins , Kelly Kelly ,
Lilian Garcia , Torrie Wilson , Michelle McCool , Terri Runnels , Maria Kanellis and the Hall of Famers Jacqueline and Trish Stratus . Other media ( edit )
Maryse and Miz in July 2016 In April 2007 , she appeared alongside fellow WWE Divas Ashley , Torrie Wilson , Brooke Adams , Layla , and Kelly Kelly in Timbaland
's music video `` Throw It On Me '' featuring The Hives . Ouellet also appeared in the January 2009 issue of Muscle & Fitness , along with Eve Torres and
Michelle McCool . She and John Morrison were interviewed on Eurosport in the same year .
</item 9>

<item 10>
Maryse Ouellet
E ! Online . Retrieved December 23 , 2017 . Jump up ^ `` Maryse Ouellet profile on famousVcard '' . Jump up ^ `` Maryse talks WWE release , restraining order ,
Divas Legend she wants to work with '' . Pro Wrestling Torch . January 5 , 2012 . Retrieved April 12 , 2012 . Jump up ^ `` More Diva dreams come true '' . WWE .
August 24 , 2006 . Retrieved June 27 , 2016 . Jump up ^ Nesbitt , Cherylann . `` Mike `` The Miz '' Mizanin Knows His Fiance is hotter `` . MTV . Retrieved
August 31 , 2013 . Jump up ^ Donnelly , Matthew Scott . `` Mike ' The Miz ' Mizanin ties the knot with a fellow WWE Champ '' . MTV . Retrieved March 21 , 2014 .
Jump up ^ `` WWE Superstar The Miz -- My Wedding Was Awesome ! '' . TMZ . February 22 , 2014 .
</item 10>
=== END ITEM id=a80c1125e1852e19 ===

=== ITEM id=cae52fd9bd83870d ===
Gold sentences and facts:
Sentence 0: The new season of The Little Couple premiered on September 19 , 2017 .
  Fact 0.0: A new season of The Little Couple premiered on September 19, 2017.

Retrieved items:
<item 1>
The Little Couple
The Little Couple - wikipedia The Little Couple Jump to : navigation , search The Little Couple Genre Reality Starring Bill Klein Jennifer Arnold - Klein
William Rijin Klein Zoey Nidhi Klein Country of origin United States Original language ( s ) English No. of seasons 8 No. of episodes 134 Production Executive
producer ( s ) Eric Schotz Ruth Rivin Melissa Tallerine Producer ( s ) Elizabeth Massie Oliver Oertel Sonya Gay Bourn Editor ( s ) Adrian Herrera Paul Yates
Ryan Ely Scott Reynolds Camera setup Multiple Running time 22 to 24 minutes Production company ( s ) LMNO Productions Release Original network TLC Picture
format 480i ( SDTV ) 1080i ( HDTV ) Original release May 26 , 2009 ( 2009 - 05 - 26 ) -- present External links Website www.thelittlecouple.net The Little
Couple is an American reality television series on TLC that debuted May 26 , 2009 . The series features Bill Klein , a businessman , and Dr. Jennifer Arnold , a
neonatologist , who both have skeletal dysplasia .
</item 1>

<item 2>
The Little Couple
2012 ) Anderson ( March 20 , 2012 ) Episodes ( edit ) Season Episodes Originally aired First aired Last aired 14 May 26 , 2009 August 4 , 2009 18 October 27 ,
2009 February 23 , 2010 19 June 1 , 2010 December 7 , 2010 24 May 31 , 2011 November 1 , 2011 5 35 March 20 , 2012 October 15 , 2013 6 16 March 4 , 2014 June 17
, 2014 7 12 December 2 , 2014 June 30 , 2015 8 10 January 5 , 2016 March 8 , 2016 Season 1 ( 2009 ) ( edit ) No. in series No. in season Title Original air date
0 0 `` Little People : Just Married '' January 19 , 2009 ( 2009 - 01 - 19 ) Bill Klein and Jennifer Arnold are introduced in this one - hour special following
them before , during , and after their wedding .
</item 2>

<item 3>
The Little Couple
6 81 `` A Store is Born '' April 3 , 2012 ( 2012 - 04 - 03 ) 0.82 Bill needs to renovate his new pet store and hire employees . Judy helps Bill interview
potential employees and Jen gives her opinion on the design . 7 82 `` A Very Klein Christmas '' April 10 , 2012 ( 2012 - 04 - 10 ) 0.87 Bill and Jen are having
a big holiday celebration with both their families in their new house . 8 83 `` Ready , Set , Relax ! '' April 10 , 2012 ( 2012 - 04 - 10 ) 0.84 Between their
recent surrogacy attempt and the busy holidays , life has become very stressful for Bill and Jen . Bill takes Jen to Galveston for a surprise weekend at the
beach . 9 84 `` Grand Opening Weekend '' April 17 , 2012 ( 2012 - 04 - 17 ) 0.68 It 's Grand Opening Weekend for Bill 's new pet store . Jen finds out she needs
to get to Los Angeles for an egg extraction during the celebration .
</item 3>

<item 4>
The Little Couple
Arnold is a neonatologist who works at the Texas Children 's Hospital , and Klein is a medical supplies and telemarketing businessman . They were originally
introduced in a one - hour TLC special , Little People : Just Married . Season 1 of the show premiered on May 26 , 2009 . Season 2 premiered on October 27 ,
2009 . On February 22 , 2010 , TLC renewed The Little Couple for a third season consisting of 20 episodes . The mid-season finale aired after 9 episodes on July
27 , 2010 , and season 3 returned on October 12 , 2010 . It was announced on Facebook that the show would be returning for its 4th season on May 31 , 2011 .
Again , the season was split , with the second half of season 4 premiering on September 27 , 2011 . In February 2012 , Klein opened Rocky&Maggie in Rice Village
Houston . named after their pets Chihuahua Rocky and rescue mutt terrier Maggie . Arnold 's mother Judy manages the store .
</item 4>

<item 5>
The Little Couple
9 135 `` New Challenges '' March 1 , 2016 ( 2016 - 03 - 01 ) N / A Jen and Bill have a night out just for adults at a local escape room . Meanwhile , Zoey has
her first Indian dance lesson ; and the family attends a Chinese lantern festival in Houston . 10 136 `` A Little 4 - 1 - 1 '' March 8 , 2016 ( 2016 - 03 - 08 )
N / A Bill and Jen answer questions and go over topics like parenting , issues from being little people , and having their own television show . References (
edit ) Jump up ^ `` TLC Heats Up Summer Schedule with New Premieres '' . The Futon Critic. April 27 , 2009 . Retrieved April 19 , 2013 . Jump up ^ Srikameswaran
, Anita ( August 7 , 2001 ) . `` Meet Dr. Arnold '' . Post-Gazette . Retrieved August 6 , 2013 . Jump up ^ George , Cindy ( May 24 , 2009 ) .
</item 5>

<item 6>
The Little Couple
15 125 `` Jen 's Big 4 - 0 '' June 10 , 2014 ( 2014 - 06 - 10 ) 2.90 Jen gets ready to turn 40 and thinks the family are going to the beach in Galveston , Texas
, but Bill has a surprise in store for her milestone birthday . 16 126 `` I Do ... Again '' June 17 , 2014 ( 2014 - 06 - 17 ) 3.06 Bill reveals the vow renewal
he has been planning and high winds put the ceremony in jeopardy . Season 7 ( 2014 -- 15 ) ( edit ) No. in season No. in series Title Original air date U.S.
viewers ( million ) 127 `` Have a Little Faith '' December 2 , 2014 ( 2014 - 12 - 02 ) 2.61 Will and Zoey are baptized in Florida . Later , the family takes a
trip to Washington , D.C.
</item 6>

<item 7>
The Little Couple
in season No. in series Title Original air date U.S. viewers ( millions ) 111 `` Zoey Makes 4 '' March 4 , 2014 ( 2014 - 03 - 04 ) 3.89 Jen and Bill go on a
trip to India to pick up their adopted daughter , Zoey , but they wonder how Will will react upon meeting his new baby sister . 112 `` The Unexpected '' March
11 , 2014 ( 2014 - 03 - 11 ) 3.16 Jen begins to feel ill on the way to New Delhi while she and Bill are in India to pick up their daughter , Zoey . She must
make a difficult decision in which she has to leave New Delhi and return to the U.S , affecting the whole family . 113 `` We 're in This Together '' March 18 ,
2014 ( 2014 - 03 - 18 ) 3.21 Jen finds out she has cancer and must go through treatment . Afterwards , the family decorates for Christmas .
</item 7>

<item 8>
The Little Couple
22 73 `` The Start of Something Big '' October 25 , 2011 ( 2011 - 10 - 25 ) 1.12 Bill and Jen ' must decide if they are willing to have their two embryos go
through genetic testing . 23 74 `` A Little More Q & A '' October 25 , 2011 ( 2011 - 10 - 25 ) 1.14 Bill and Jen take viewer questions and an inside view of the
past four seasons . 24 75 `` Are We Pregnant ? '' November 1 , 2011 ( 2011 - 11 - 01 ) 1.50 Bill and Jen 's embryos have thawed and now it 's time to transfer
them to their surrogate . Season 5 ( 2012 -- 13 ) ( edit ) No. in series No. in season Title Original air date U.S. viewers ( millions ) 76 `` Getting Back on
Track '' March 20 , 2012 ( 2012 - 03 - 20 ) 1.11 Bill and Jen are ready to move forward following their surrogate 's miscarriage .
</item 8>

<item 9>
The Little Couple
12 138 `` A Little Q&A '' June 30 , 2015 ( 2015 - 06 - 30 ) 1.44 Jen and Bill answer questions from their fans about their lives and feelings . Season 8 ( 2016
) ( edit ) No. in season No. in series Title Original air date U.S. viewers ( million ) 127 `` Zoey 's 4th Birthday '' January 5 , 2016 ( 2016 - 01 - 05 ) 2.61
In the Season 10 premiere , Zoey 's about to turn 4 , and Jen decides on a princess tea party theme for her celebration , but Bill worries that her planning is
a bit over the top . 128 `` Back to School '' January 12 , 2016 ( 2016 - 01 - 12 ) 2.09 The kids get haircuts as the new school year approaches . Also : Bill
takes Will grocery shopping for his lunches ; and Jen and Zoey go for a mani - pedi . 129 `` Safety First ! ''
</item 9>

<item 10>
The Little Couple
`` ' The Little Couple ' Season 6 to Premiere Tuesday , March 4 on TLC '' . TV by the Numbers ( Press release ) . Retrieved February 5 , 2014 . Jump up ^ ``
Tuesday 's Cable Ratings : 10 - 19 - 10 '' . The Futon Critic . Jump up ^ `` Tuesday 's Cable Ratings : 10 - 26 - 10 '' . The Futon Critic . Jump up ^ ``
Tuesday 's Cable Ratings : 11 - 2 - 10 '' . The Futon Critic . Jump up ^ `` Tuesday 's Cable Ratings : 11 - 9 - 10 '' . The Futon Critic . Jump up ^ `` Tuesday
's Cable Ratings : 11 - 23 - 10 '' . The Futon Critic . Jump up ^ `` Tuesday 's Cable Ratings : 11 - 30 - 10 '' . The Futon Critic . ^ Jump up to : `` Tuesday
's Cable Ratings : 5 - 31 - 11 '' . The Futon Critic .
</item 10>
=== END ITEM id=cae52fd9bd83870d ===

=== ITEM id=53840d110cbde13c ===
Gold sentences and facts:
Sentence 0: A transfer case is a part of the drivetrain of four - wheel - drive , all - wheel - drive , and other multiple powered axle vehicles . The transfer case transfers power from the transmission to the front and rear axles by means of drive shafts . It also synchronizes the difference between the rotation of the front and rear wheels , and may contain one or more sets of low range gears for off - road use .
  Fact 0.0: A transfer case is a part of the drivetrain of four-wheel-drive, all-wheel-drive, and other multiple powered axle vehicles.
  Fact 0.1: A transfer case transfers power from the transmission to the front and rear axles by means of drive shafts.
  Fact 0.2: A transfer case synchronizes the difference between the rotation of the front and rear wheels.
  Fact 0.3: A transfer case may contain one or more sets of low range gears for off-road use.

Retrieved items:
<item 1>
Transfer case
See also ( edit ) NP - 205 AMC / Jeep Transmissions Jeep four wheel drive systems Mitsubishi Super Select References ( edit ) Jump up ^
http://www.cdxetextbook.com/trans/finalDrives/allWheel/transfercasediff.html Transfer case differential action Jump up ^
http://www.fourwheeler.com/techarticles/68958_how_transfer_cases_work/index.html How Transfer Cases Work Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Transfer_case&oldid=805616544 '' Categories : Vehicle technology Four - wheel drive layout Hidden categories : All
articles with unsourced statements Articles with unsourced statements from May 2015 Articles with unsourced statements from March 2012 Talk Contents About
Wikipedia فارسی 日本 語 Polski Русский Simple English Svenska Українська Edit links This page was last edited on 16 October 2017 , at 15 : 12 . Text is available
under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy
.
</item 1>

<item 2>
Transfer case
Those used in sports cars and performance sedans are usually `` transparent '' to the driver ; there is no shifter or select lever . Drive type ( edit ) Gear -
driven ( edit ) There are two different types of internal power - transfer mechanism found in most transfer cases . Gear - driven transfer cases use sets of
gears to drive either the front or both the front and rear driveshafts . These are generally strong , heavy units that are used in large trucks , but there are
currently several gear drive cases in production for passenger cars . Chain - driven ( edit ) Chain - driven transfer cases use a chain to drive most often only
one axle , but can drive both axles . Chain - driven transfer cases are quieter and lighter than gear - driven ones . They are used in vehicles such as compact
trucks , full size trucks , Jeeps and SUVs . Some off - road driving enthusiasts modify their vehicles to use gear - driven transfer cases , accepting the
additional weight and noise to gain the extra strength they generally provide . Housing type ( edit ) Married ( edit ) Transfer cases are also classified as
either `` divorced '' / independent or `` married '' .
</item 2>

<item 3>
Transfer case
This is only for the four - wheel - drive high setting . To engage the four - wheel - drive low setting , the vehicle must be stopped and the transmission must
be shifted to neutral , then the four - wheel - drive low can be selected . ESOF ( edit ) Electronic Shift On - the - Fly ( ESOF ) transfer cases have a dash -
mounted selector switch or buttons with front sealed automatic locking axle hubs or drive flanges . Some vehicles , such as Jeeps , also have what is called
SelecTrac , which has a slider switch on the center console . Unlike the manual transfer case , this system has a transfer case motor . To engage the four -
wheel - drive system the vehicle must be moving at a lower speeds , the speed at which 4x4 can be engaged depends on the vehicle . This is only for the four -
wheel - drive high setting . To engage the four - wheel - drive low setting , the vehicle must be stopped and the transmission must be shifted to neutral , then
the four - wheel - drive low can be selected .
</item 3>

<item 4>
Transfer case
Married transfer cases are bolted directly to the transmission , usually between the transmission 's output shaft and the rear or main driveshaft . Sometimes a
married transfer case is an integral part of the transmission and the two components share the same housing or `` case '' , as is commonly found on recent
Subaru products and some other all - wheel - drive cars . Divorced / independent ( edit ) A divorced or independent transfer case is completely separate from
the transmission . It is located further down the driveline than a married transfer case and connected to the transmission output shaft by a short driveshaft .
Independent transfer cases are used on very long wheelbase vehicles , such as commercial trucks or military trucks . Transfer case Shift type ( edit ) M.S.O.F.
( edit ) Manual Shift On - the - Fly transfer cases have a selector lever on the driver 's side floor transmission hump and may also have either two sealed
automatic front axle locking hubs or two manual front axle hub selectors of `` LOCK '' and `` UNLOCK '' or `` FREE '' . To engage the four - wheel - drive
system the vehicle must be moving at a low speed , the speed at which 4x4 can be engaged depends on the vehicle .
</item 4>

<item 5>
Transfer case
This is necessary because the front and rear tires never turn at the same speed . Different rates of tire rotation are generally due to different tire diameters
( since front and rear tires inevitably wear at different rates ) and different gear ratios in the front and rear differentials since manufacturers will often
have a slightly lower ratio in the front vs. the rear to help with control ( such as a 3.55 : 1 in the rear differential and a 3.54 : 1 in the front
differential ) . If the transfer case did not make up the difference between the two different rates of rotation , binding would occur and the transfer case
could become damaged . This is also why a transfer case that is not designed for on - road use must never be driven on dry pavement . Transfer cases designed
for off - road use can mechanically lock the front and rear axles when needed ( e.g. when one of the axles is on a slippery surfaces or stuck in mud , whereas
the other has better traction ) . This is the equivalent to the differential lock . The transfer case may contain one or more sets of low range gears for off -
road use .
</item 5>

<item 6>
Transfer case
1 M.S.O.F. 2.3. 2 E.S.O.F. 3 See also 4 References Functions ( edit ) The transfer case receives power from the transmission and sends it to both the front and
rear axles . This can be done with gears , hydraulics , or chain drive . On some vehicles , such as four - wheel - drive trucks or vehicles intended for off -
road use , this feature is controlled by the driver . The driver can put the transfer case into either `` two - wheel - drive '' or `` four - wheel - drive ''
mode . This is sometimes accomplished by means of a shifter , similar to that in a manual transmission . On some vehicles this may be electronically operated by
a switch instead . Some vehicles , such as all - wheel - drive sports cars , have transfer cases that are not selectable . Such a transfer case is permanently
`` locked '' into all - wheel - drive mode . Transfer cases that are designed to allow for normal road use synchronize the difference between the rotation of
the front and rear wheels , in much the same way the differential acts on a given axle . This is necessary because the front and rear tires never turn at the
same speed .
</item 6>

<item 7>
Jeep
Off - road abilities Jeep Wrangler off - roading . Jeep advertising has always emphasized the brand 's vehicles ' off - road capabilities . Today , the Wrangler
is one of the few remaining four - wheel - drive vehicles with solid front and rear axles . These axles are known for their durability , strength , and
articulation . New Wranglers come with a Dana 44 rear differential and a Dana 30 front differential . The upgraded Rubicon model of the JK Wrangler is equipped
with electronically activated locking differentials , Dana 44 axles front and rear with 4.10 gears , a 4 : 1 transfer case , electronic sway bar disconnect and
heavy duty suspension . Another benefit of solid axle vehicles is they tend to be easier and cheaper to `` lift '' with aftermarket suspension systems . This
increases the distance between the axle and chassis of the vehicle . By increasing this distance , larger tires can be installed , which will increase the
ground clearance , allowing it to traverse even larger and more difficult obstacles . In addition to higher ground clearance , many owners aim to increase
suspension articulation or `` flex '' to give their Jeeps greatly improved off - road capabilities .
</item 7>

<item 8>
Transfer case
Transfer case - wikipedia Transfer case Jump to : navigation , search For the US military use of the term , see coffin . Inside of a 231 New Process Gear
transfer case . Part - time / Manual , shift on the fly A transfer case is a part of the drivetrain of four - wheel - drive , all - wheel - drive , and other
multiple powered axle vehicles . The transfer case transfers power from the transmission to the front and rear axles by means of drive shafts . It also
synchronizes the difference between the rotation of the front and rear wheels , and may contain one or more sets of low range gears for off - road use .
Contents ( hide ) 1 Functions 2 Types 2.1 Drive type 2.1. 1 Gear - driven 2.1. 2 Chain - driven 2.2 Housing type 2.2. 1 Married 2.2. 2 Divorced / independent
2.3 Transfer case shift type 2.3. 1 M.S.O.F. 2.3. 2 E.S.O.F.
</item 8>

<item 9>
Transfer case
The transfer case may contain one or more sets of low range gears for off - road use . Low range gears are engaged with a shifter or electronic switch . On many
transfer cases , this shifter is the same as the one that selects 2WD or 4WD operation . Low range gears allow the vehicle to drive at much slower speeds while
still operating within the usable power band / RPM range of the engine . This also increases the torque available at the axles . Low - range gears are used
during slow - speed or extreme off - road maneuvers , such as rockcrawling , navigating dangerous roads , or when pulling a heavy load . This feature is often
absent on all - wheel - drive cars . Some very large vehicles , such as heavy equipment or military trucks , have more than one low - range gear . Types ( edit
) Transfer cases used on `` part - time '' four - wheel - drive off - road vehicles such as trucks , truggies , rock - crawling vehicles , and some military
vehicles generally allow the driver to select 2WD or 4WD , as well as high or low gear ranges .
</item 9>

<item 10>
Transfer case
By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit
organization . About Wikipedia
</item 10>
=== END ITEM id=53840d110cbde13c ===

=== ITEM id=b7572fbf0fba5aa1 ===
Gold sentences and facts:
Sentence 0: No . Name Province Birth / death Term of active service Term of service as Chief Length in days of active service Length in days as Chief Appointed by Law school Position prior to appointment Codrington Edmund Carrington England 1769 - 1849 March 1801 -- 2 April 1806 March 1801 -- 2 April 1806 North -- Counsel to the East India Company Edmund Henry Lushington England 1766 - 1839 5 September 1801 -- 1809 15 April 1807 - 1809 North University of Cambridge Barrister Alexander Johnston Scotland 1775 - 1849 15 April 1807 -- 14 February 1809 6 November 1811 - 1819 6 November 1811 - 1819 671 + ? Maitland University of Göttingen Advocate Fiscal of Ceylon William Coke England 1776 - 1818 6 March 1809 -- 1 September 1818 † 6 March 1809 -- 1811 3,466 Maitland University of Oxford Advocate Fiscal of Ceylon 5 Henry Byrne 5 November 1818 -- 1819 -- -- Brownrigg Barrister 6 Ambrose Hardinge Giffard Ireland 1771 -- 1827 8 April 1819 -- 2 March 1827 8 April 1819 -- 2 March 1827 2,885 2,885 Brownrigg Trinity College , Dublin Advocate Fiscal of Ceylon 7 Richard Ottley 1782 -- 1845 8 May 1820 -- 1833 1 November 1827 -- 1833 Barnes Chief Justice of Grenada 8 Henry Matthews England 1789 -- 1828 10 October 1827 -- 1828 -- -- Barnes King 's College , Cambridge Advocate Fiscal of Ceylon 9 Charles Marshall England 1788 - 1873 14 June 1828 -- 3 March 1836 18 February 1833 -- 3 March 1836 2,850 1,109 Barnes Jesus College , Cambridge 10 William Rough England 1772 - 1838 18 February 1833 - 1838 -- -- Trinity College , Cambridge 11 William Norris 1834 -- 1837 27 April 1836 -- 1837 Wilmot - Horton Advocate Fiscal of Ceylon 12 John Jeremie Guernsey 1795 - 1841 10 March 1836 -- 1840 -- -- Wilmot - Horton Public prosecutor of Mauritius 13 John Fredrick Stoddart Malta 1805 - 1839 2 June 1838 -- 19 August 1839 † -- 443 -- University of Glasgow 14 Anthony Oliphant Scotland 1793 -- 1859 22 October 1838 - 1854 22 October 1838 - 1854 Attorney General of the Cape Colony 15 J.G. Hildebrand 23 November 1839 -- -- -- 16 William Ogle Carr England 1802 - 1856 19 December 1939 - 1856 17 April 1854 -- 1856 Stewart - Mackenzie King 's Advocate of Ceylon 17 John Stark 10 October 1840 -- -- -- Queen 's Advocate of Ceylon 18 Christopher Temple 28 January 1854 -- -- -- Judge of the District Court of Colombo 19 Paul Ivy Sterling 20 November 1855 -- -- -- Attorney General of Hong Kong 20 William Carpenter Rowe - 1860 1857 -- 1859 -- 21 Edward Shepherd Creasy England 1812 - 1878 -- 27 March 1860 - 1875 University of Cambridge 22 Richard Morgan -- -- -- Queen 's Advocate of Ceylon 23 George Lawson -- -- -- 24 Henry Byerley Thompson - 1867 1 January 1863 -- -- -- Queen 's Advocate of Ceylon 25 Charles Henry Stewart 11 January 1867 -- -- -- 26 Richard Cayley England 26 July 1874 -- 1882 1 October 1879 -- 1882 -- Queen 's Advocate of Ceylon 27 Lovell Burchett Clarence 5 July 1879 -- -- -- 28 Harry Dias Bandaranaike 5 July 1879 -- -- -- 29 William Hackett 3 February 1877 - 1877 3 February 1877 - 1877 -- 30 John Budd Phear England 18 October 1877 -- -- -- 31 Archibald Campbell Lawrie 1837 - 1914 6 July 1892 -- 1901 -- -- Judge of the District Court of Kandy 32 Jacobus de Wet South Africa 31 May 1882 -- 29 May 1883 31 May 1882 -- 29 May 1883 363 363 33 Bruce Burnside Bahamas 21 May 1883 -- 1893 21 May 1883 -- 1893 -- Queen 's Advocate of Ceylon 34 John Winfield Bonser 13 November 1893 -- 1902 13 November 1893 -- 1902 Chief Justice of the Straits Settlements 35 George Henry Withers -- -- -- 36 Dodwell Browne -- -- -- 37 Fredrick Charles Moncrieff 4 April 1900 -- -- -- 38 Henry Lorentz Wendt 15 October 1901 -- 28 November 1910 -- -- 39 John Page Middleton 28 January 1902 -- 1912 -- -- Puisine Judge of the Supreme Court of Cyprus 40 Charles Layard Western 26 April 1902 -- 18 June 1906 26 April 1902 -- 18 June 1906 1,514 1,514 41 Joseph Richard Grenier 28 November 1910 -- 1912 -- -- 42 Alexander Wood Renton 3 September 1905 -- 1918 22 August 1914 - 1918 -- 43 Alfred George Lascelles England 1 May 1911 -- 1914 1 May 1911 -- 1914 -- 44 Joseph Turner Hutchinson England 23 October 1906 -- 1 May 1911 23 October 1906 -- 1 May 1911 1,651 1,651 45 James Cecil Walter Pereira 4 July 1912 -- -- -- 46 James Arthur Van Langenburg -- -- -- 47 Thomas De Sampayo 1915 -- -- -- University of Cambridge Commissioner of Assize 48 George Francis Macdaniel Ennis 5 August 1912 -- -- -- 49 Walter Shaw 10 December 1914 -- -- -- Chief Justice of British Honduras 50 Gualterus Stewart Schneider -- -- -- 51 Anton Bertram 1869 - 1937 26 July 1918 -- 1925 26 July 1918 -- 1925 University of Cambridge Attorney General of Ceylon 52 Hermann Albert Loos -- -- -- 53 William Thomas Porter 12 January 1922 -- -- -- 54 Thomas Garvin 1881 - 1940 21 June 1921 -- 1935 -- -- Manning Colombo Law College Solicitor General of Ceylon 55 John Adrian St. Valentine Jayewardene 1877 -- 1927 1923 -- -- -- Manning Judge of the District Court of Colombo 56 Llwellyn Chisholm Dalton 14 March 1925 -- -- -- 57 Lewis Matthew Maartensz 18 February 1935 -- -- -- 58 Maas Thajoon Akbar 1880 - 1944 -- -- -- University of Cambridge Solicitor General of Ceylon 59 Charles Ernest St. John Branch 1865 - 1939 3 July 1925 -- 25 May 1926 3 July 1925 -- 25 May 1926 326 326 Durham University 60 Stanley Fisher 1867 - 11 December 1926 -- 1930 11 December 1926 -- 1930 Chief Justice of Trinidad and Tobago 61 Robert William Lyall Grant 17 February 1926 -- -- -- 62 Allen Drieberg 19 September 1927 -- -- -- 63 Eugene Wilfred Jayewardene Western 1874 - 1932 May 1928 -- -- -- Fletcher Member of the Colombo Municipal Council 64 Philip James Macdonell 1873 - 1940 3 October 1930 - 1936 3 October 1930 - 1936 Chief Justice of Trinidad and Tobago 65 Lucian Macull Dominic de Silva -- -- -- 66 William Edward Barber -- -- -- 67 Francis Harold Bertram 19 February 1935 -- -- -- 68 Kenneth Poyser -- -- -- 69 Francis Soertsz - 1951 12 November 1936 -- -- -- 70 Sidney Abrahams England 1885 - 1957 3 July 1936 -- 1939 3 July 1936 -- 1939 University of Cambridge 71 Francis Moseley 2 August 1936 -- -- -- 72 Vincent Fernando 11 November 1937 -- -- -- 73 Hector Hearne 21 May 1937 -- -- -- 74 Arthur Eric Keuneman 4 March 1938 -- -- -- University of Cambridge 75 Oswald Leslie De Kretser II Western 1882 - 1959 8 April 1938 -- 1945 -- -- Caldecott University College , Colombo Commissioner of Assize 76 Arthur Wijewardena 1887 - 1964 22 August 1938 -- 1950 15 January 1949 -- 1950 University of Cambridge Solicitor General of Ceylon 77 J.H.B. Nihill 18 November 1938 -- -- -- 78 John Curtois Howard 1 December 1939 -- 1945 1 December 1939 -- 1945 Attorney General of Cyprus 79 G.H.F. Cannon -- -- -- 80 Edward Jayetileke 14 May 1942 -- 11 October 1951 1950 -- 11 October 1951 3,437 Caldecott Solicitor General of Ceylon 81 Alan Rose 11 January 1945 -- 1956 11 October 1951 -- 1956 82 A.R.H. Canakeratne 1886 - 1960 28 March 1945 -- -- -- Ceylon Law College University of London Commissioner of Assize 83 Manikku Wadumestri Hendrick de Silva -- -- -- Attorney General of Ceylon 84 Felix Reginald Dias Bandaranaike I 1861 - 1947 3 September 1946 -- -- -- University of Cambridge Judge of the District Court of Colombo 85 Chellappah Nagalingam 1893 - 1958 22 July 1947 -- 1958 -- -- Ceylon Law College Attorney General of Ceylon 86 Ralph Windham 23 July 1947 -- -- -- 87 Hema Henry Basnayake 1902 - 23 October 1947 -- 3 August 1964 1 January 1956 -- 3 August 1964 6,129 3,137 Senanayake Attorney General of Ceylon 88 Edward Frederick Noel Gratiaen 22 March 1948 -- -- -- Senanayake Attorney General of Ceylon 89 E.H.T. Gunasekera 1 March 1949 -- 25 January 1962 -- 4,713 -- Senanayake 90 M.F.S. Pulle 28 September 1949 -- -- -- Senanayake 91 V.L. St. C. Swan 1 May 1950 -- -- -- Senanayake 92 H.A. de Silva 30 November 1950 -- -- -- Senanayake 93 Kairshasp Choksy Western -- -- -- 94 Henry Weerasooriya 10 July 1953 -- 24 February 1964 -- 3,881 -- Senanayake 95 K.D. de Silva 10 July 1953 -- -- -- Senanayake 96 Miliani Sansoni 10 May 1954 -- 17 November 1966 3 August 1964 -- 17 November 1966 4,574 836 Kotelawala 97 Hugh Fernando 1910 - 1976 16 May 1955 -- 17 November 1973 20 November 1966 - 17 November 1973 6,760 2,554 Kotelawala 98 Thusew Samuel Fernando 2 May 1956 -- -- -- Bandaranaike Colombo Law College Solicitor General of Sri Lanka 99 Nadaraja Sinnetamby 3 May 1956 -- -- -- Bandaranaike 100 Lionel de Silva 5 November 1959 -- -- -- Dahanayake 101 Leonard de Silva 16 August 1966 -- -- -- Senanayake 102 Henry Thambiah Northern 1906 -- 1997 1 December 1961 -- 27 May 1968 -- 2,369 -- Bandaranaike University College , Colombo Ceylon Law College University of London Commissioner of Assize 103 Kingsley Herath 25 January 1962 -- -- -- Bandaranaike 104 Asoka Abeyesundere Southern 1906 - 1988 20 April 1962 -- 3 May 1968 -- 2,205 -- Bandaranaike Ceylon Law College 105 Ponnuduraiswamy Sriskandarajah 1905 - 1968 22 June 1962 -- 8 February 1967 -- 1,692 -- Bandaranaike Ceylon University College Ceylon Law College Commissioner of Assize 106 Gardiye Punchihewage Amaraseela Silva 22 June 1962 -- 1974 1973 -- 1974 -- Bandaranaike 107 Lionel Sirimanne -- -- -- Bandaranaike 108 Vaithalingam Manicavasagar Western 1906 - 1993 17 December 1964 -- 14 December 1967 -- 1,092 -- Bandaranaike Colombo Law College Commissioner of Assize 109 Anthony Alles Central 1911 -- 2003 1964 -- 1974 -- -- Bandaranaike University of Ceylon Ceylon Law College University of London Solicitor General of Sri Lanka 110 Veeravagu Sivasubramaniam 1908 - 1985 1966 -- 1970 -- -- Senanayake University of London Commissioner of Assize 111 George Samerawickrame 18 November 1966 -- 9 April 1981 -- 5,256 -- Senanayake 112 Victor Tennekoon Central 1914 - 8 February 1967 -- 1 July 1970 1 January 1974 - 8 September 1977 1 January 1974 - 8 September 1977 1,239 + 1,346 1,346 Senanayake University of London Ceylon Law College Judge of the Court of Appeal 113 Christopher Weeramantry Western 1926 - 1967 -- 30 April 1972 -- -- Senanayake University of Ceylon University of London Commissioner of Assize 114 Oswald Leslie De Kretser III Western 1910 - 1968 -- 21 January 1972 -- -- Senanayake University College , Colombo University of London Judge of the District Court of Panadura 115 Samarapilimudalige Wijayatilake Central 1912 - 1990 1968 -- 16 May 1975 -- -- Senanayake University College , Colombo Ceylon Law College Commissioner of Assize 116 Pandita Gunawadena 1968 -- 10 February 1971 -- -- Senanayake 117 Vincent Thamotheram 1915 - 23 March 1970 -- 1980 -- -- Senanayake Ceylon University College Ceylon Law College Commissioner of Assize 118 Hector Deheragoda 22 February 1972 -- 27 December 1976 -- 1,770 -- Bandaranaike 119 Cuda Banda Walgampaya 22 February 1972 -- 3 January 1976 -- 1,411 -- Bandaranaike 120 Jaya Pathirana - 2000 22 February 1972 -- 1978 -- -- Bandaranaike 121 Dharmadasa Wimalaratne 10 April 1972 - ? 20 October 1980 -- 15 September 1985 -- 1,791 -- Bandaranaike 122 Tellipalai Rajaratnam 1920 - 1994 10 April 1972 -- 1978 -- -- Bandaranaike Commissioner of Assize 123 Donald Sirimanne 1973 -- -- -- Gopallawa 124 Cyril Udalagama 1 January 1974 -- -- -- Gopallawa 125 T.A. de S. Wijesundera 1 January 1974 -- -- -- Gopallawa 126 Malcolm Perera 1 January 1974 -- -- -- Gopallawa 127 Izadean Mohamed Ismail 1 January 1974 -- 18 December 1981 -- 2,908 -- Gopallawa 128 Joe Weereratne 1 January 1974 -- -- -- Gopallawa 129 Arulampalam Vythialingam 1 January 1974 -- -- -- Gopallawa 130 Noel Tittawela 1 January 1974 -- -- -- Gopallawa 131 Suppiah Sharvananda Northern 1923 - 2007 1 January 1974 -- 1988 29 October 1984 -- 1988 Gopallawa Colombo Law College University of London 132 Somaratne Walpita 1 January 1974 -- -- -- Gopallawa 133 Wilmot Gunasekera 2 May 1974 -- -- -- Gopallawa 134 Barnes Ratwatte II 1918 - 2004 2 December 1974 -- ? 11 April 1981 -- 22 November 1983 -- 955 -- Gopallawa Jayewardene Ceylon Law College President of the Court of Appeal of Sri Lanka 135 Raja Wanasundera Sabaragamuwa 1924 - 2012 5 June 1975 -- ? -- -- Gopallawa University College , Colombo Ceylon Law College University of London Stanford University Solicitor General of Sri Lanka 136 Percy Colin - Thomé Southern 1922 - 12 January 1976 -- 1977 29 December 1981 -- ? -- -- Gopallawa Jayewardene University of Ceylon University of Cambridge Judge of the High Court of Colombo 137 Neville Samarakoon 1919 - 1990 -- 21 October 1984 1977 -- 21 October 1984 Jayewardene University College , Colombo Colombo Law College 138 Victor Perera 24 July 1981 -- 18 August 1983 -- 755 -- Jayewardene 139 Anton Soza Western 1919 - 2013 26 January 1982 -- 3 February 1984 -- 738 -- Jayewardene University of London Judge of the Court of Appeal 140 Parinda Ranasinghe 15 November 1982 -- 1991 1988 -- 1991 Jayewardene Colombo Law College 141 Abdul Cader 8 December 1982 -- 28 May 1985 -- 902 -- Jayewardene 142 Heevi Rodrigo 8 December 1982 -- 4 December 1984 -- 727 -- Jayewardene 143 E.A.D. Atukorale 27 November 1984 -- -- -- Jayewardene 144 Herbert Thambiah Northern 1926 -- 1992 27 November 1984 -- 1992 1991 -- 1991 Jayewardene University of Ceylon Ceylon Law College Judge of the Court of Appeal 145 Lucian Hector de Alwis 27 November 1984 -- -- -- Jayewardene 146 O.S.M. Seneviratne 10 February 1986 -- -- -- Jayewardene 147 H.A.G. de Silva 10 February 1986 -- -- -- Jayewardene Mark Fernando 1941 - 2009 4 March 1988 -- 2005 -- -- Jayewardene University of Ceylon , Peradeniya A.R.B. Amerasinghe 1937 - 2013 -- -- -- A.R.N. Fernando -- -- -- Ameer Ismail 1990 -- 1994 -- -- Premadasa University of Ceylon Ceylon Law College President of the Court of Appeal of Sri Lanka Andrew Ranjan Perera -- -- -- Pathmanathan Ramanathan 1932 - 2006 -- -- -- University of Wales President of the Court of Appeal of Sri Lanka K.M.M.B. Kulatunge -- -- -- G.P.S. de Silva -- 1991 -- 1999 Sarath N. Silva 1995 -- 16 September 1999 16 September 1999 -- 7 June 2009 3,552 Kumaratunga Sri Lanka Law College University of Brussels President of the Court of Appeal of Sri Lanka Shirani Bandaranayake North Western 1958 - 30 October 1996 -- 13 January 2013 28 January 2015 -- 29 January 2015 18 May 2011 -- 13 January 2013 28 January 2015 -- 29 January 2015 5,919 + 1 606 + 1 Kumaratunga University of Colombo University of London Lawyer C.V. Vigneswaran Western 1939 - March 2001 -- October 2004 -- -- Kumaratunga University of Ceylon Ceylon Law College Judge of the Court of Appeal Asoka de Silva 1946 - 1 August 2001 -- 17 May 2011 8 June 2009 -- 17 May 2011 3,576 708 Kumaratunga University of Peradeniya University of Colombo University of Illinois at Urbana -- Champaign President of the Court of Appeal of Sri Lanka Nihal Jayasinghe 19 February 2003 -- 29 February 2008 -- 1,836 -- Kumaratunga University of Colombo Judge of the Court of Appeal Nimal Dissanayake -- 23 April 2008 -- -- Kumaratunga Shiranee Tilakawardane 19 February 2003 - 5 April 2014 -- 1,901 -- Kumaratunga Sri Lanka Law College President of the Court of Appeal of Sri Lanka Nimal Gamini Amaratunga January 2005 -- March 2014 -- -- Kumaratunga University of Colombo International Maritime Law Institute Judge of the Court of Appeal Saleem Marsoof February 2005 -- 2014 -- -- Kumaratunga University of Colombo , University of San Diego School of Law President of the Court of Appeal of Sri Lanka Andrew Somawansa North Western 28 May 2006 -- October 2008 -- -- Rajapaksa University of Ceylon University of Nevada President of the Court of Appeal of Sri Lanka Jagath Balapatabendi Southern -- -- -- Rajapaksa Ceylon Law College Judge of the Court of Appeal Kanagasabapathy Sripavan † 1952 -- 27 March 2008 -- present 30 January 2015 -- 28 February 2017 7003368200000000000 ♠ 3,682 7003118200000000000 ♠ 1,182 Rajapaksa Sri Lanka Law College Judge of the Court of Appeal P.A. Ratnayake 2 May 2008 -- 15 February 2013 -- 7003175100000000000 ♠ 1,751 -- Rajapaksa University of Brussels Chandra Ekanayake November 2008 -- 2016 -- -- Rajapaksa Sri Lanka Law College President of the Court of Appeal of Sri Lanka S.I. Imam Central 9 January 2009 -- 20 February 2013 -- 1,503 -- Rajapaksa Judge of the Court of Appeal R.K.S. Suresh Chandra 17 June 2010 -- -- -- Rajapaksa Sathya Hettige 10 July 2011 -- 18 April 2014 -- 1,013 -- Rajapaksa University of Colombo President of the Court of Appeal of Sri Lanka Priyasath Dep † 10 July 2011 -- present 02 March 2017 - present -- 7003248200000000000 ♠ 2,482 -- Rajapaksa University of Colombo International Institute of Social Studies Solicitor General of Sri Lanka Eva Wansundera † North Western 7 July 2012 -- present -- 7003211900000000000 ♠ 2,119 -- Rajapaksa Sri Lanka Law College University of Leicester Attorney General of Sri Lanka Mohan Peiris ♠ 15 January 2013 -- 28 January 2015 15 January 2013 -- 28 January 2015 743 743 Rajapaksa Sri Lanka Law College Attorney General of Sri Lanka Rohini Marasinghe 26 April 2013 -- 2015 -- -- Rajapaksa University of Hartford Judge of the Court of Appeal Buwaneka Aluwihare † 4 December 2013 -- present -- 7003160400000000000 ♠ 1,604 -- Rajapaksa University of London Additional Solicitor General of Sri Lanka Sisira de Abrew † Western 7 May 2014 -- present -- 7003145000000000000 ♠ 1,450 -- Rajapaksa Sri Lanka Law College President of the Court of Appeal of Sri Lanka Sarath de Abrew 1953 -- 2016 7 May 2014 -- January 2016 -- -- Rajapaksa Sri Lanka Law College Judge of the Court of Appeal Priyantha Jayawardena † 7 May 2014 -- present -- 7003145000000000000 ♠ 1,450 -- Rajapaksa Sri Lanka Law College University of Aberdeen Upali Abeyratne † 17 December 2014 -- present -- 7003122600000000000 ♠ 1,226 -- Rajapaksa Judge of the Court of Appeal Anil Goonaratne † 30 January 2015 -- present -- 7003118200000000000 ♠ 1,182 -- Sirisena Sri Lanka Law College University of Exeter Judge of the Court of Appeal Kankanithanthri Chitrasiri † 3 December 2015 -- present -- 7002875000000000000 ♠ 875 -- Sirisena University of Sri Lanka Queen Mary University of London Judge of the Court of Appeal Nalin Perera † 3 March 2016 -- present -- 7002784000000000000 ♠ 784 -- Sirisena Judge of the Court of Appeal Prasanna Sujeewa Jayawardena † 16 June 2016 -- present -- 7002679000000000000 ♠ 679 -- Sirisena University of Colombo Judge of the Court of Appeal
  Fact 0.0: Edmund Carrington was from England, born 1769 and died 1849.
  Fact 0.1: Edmund Carrington's active service term was March 1801 to 2 April 1806.
  Fact 0.2: Edmund Carrington's province was North.
  Fact 0.3: Edmund Carrington's prior position was counsel to the East India Company.
  Fact 0.4: Edmund Henry Lushington was from England, born 1766 and died 1839.
  Fact 0.5: Edmund Henry Lushington's active service term began 5 September 1801 and ended in 1809.
  Fact 0.6: Edmund Henry Lushington's term as chief was 15 April 1807 to 1809.
  Fact 0.7: Edmund Henry Lushington's province was North.
  Fact 0.8: Edmund Henry Lushington studied at the University of Cambridge.
  Fact 0.9: Edmund Henry Lushington's prior position was barrister.
  Fact 0.10: Alexander Johnston was from Scotland, born 1775 and died 1849.
  Fact 0.11: Alexander Johnston's active service term was 15 April 1807 to 14 February 1809.
  Fact 0.12: Alexander Johnston's term as chief was 6 November 1811 to 1819.
  Fact 0.13: Alexander Johnston was appointed by Maitland.
  Fact 0.14: Alexander Johnston studied at the University of Göttingen.
  Fact 0.15: Alexander Johnston's prior position was Advocate Fiscal of Ceylon.
  Fact 0.16: William Coke was from England, born 1776 and died 1818.
  Fact 0.17: William Coke's active service term was 6 March 1809 to 1 September 1818.
  Fact 0.18: William Coke's term as chief was 6 March 1809 to 1811.
  Fact 0.19: William Coke was appointed by Maitland.
  Fact 0.20: William Coke studied at the University of Oxford.
  Fact 0.21: William Coke's prior position was Advocate Fiscal of Ceylon.
  Fact 0.22: Henry Byrne was chief from 5 November 1818 to 1819.
  Fact 0.23: Henry Byrne was appointed by Brownrigg.
  Fact 0.24: Henry Byrne's prior position was barrister.
  Fact 0.25: Ambrose Hardinge Giffard was from Ireland, born 1771 and died 1827.
  Fact 0.26: Ambrose Hardinge Giffard's active service term was 8 April 1819 to 2 March 1827.
  Fact 0.27: Ambrose Hardinge Giffard was appointed by Brownrigg.
  Fact 0.28: Ambrose Hardinge Giffard studied at Trinity College, Dublin.
  Fact 0.29: Ambrose Hardinge Giffard's prior position was Advocate Fiscal of Ceylon.
  Fact 0.30: Richard Ottley lived from 1782 to 1845.
  Fact 0.31: Richard Ottley's active service term was 8 May 1820 to 1833.
  Fact 0.32: Richard Ottley's term as chief was 1 November 1827 to 1833.
  Fact 0.33: Richard Ottley was appointed by Barnes.
  Fact 0.34: Richard Ottley's prior position was Chief Justice of Grenada.
  Fact 0.35: Henry Matthews was from England, born 1789 and died 1828.
  Fact 0.36: Henry Matthews's active service term was 10 October 1827 to 1828.
  Fact 0.37: Henry Matthews was appointed by Barnes.
  Fact 0.38: Henry Matthews studied at King's College, Cambridge.
  Fact 0.39: Henry Matthews's prior position was Advocate Fiscal of Ceylon.
  Fact 0.40: Charles Marshall was from England, born 1788 and died 1873.
  Fact 0.41: Charles Marshall's active service term was 14 June 1828 to 3 March 1836.
  Fact 0.42: Charles Marshall's term as chief was 18 February 1833 to 3 March 1836.
  Fact 0.43: Charles Marshall was appointed by Barnes.
  Fact 0.44: Charles Marshall studied at Jesus College, Cambridge.
  Fact 0.45: William Rough was from England, born 1772 and died 1838.
  Fact 0.46: William Rough's term was 18 February 1833 to 1838.
  Fact 0.47: William Rough studied at Trinity College, Cambridge.
  Fact 0.48: William Norris's term was 27 April 1836 to 1837.
  Fact 0.49: William Norris was appointed by Wilmot-Horton.
  Fact 0.50: William Norris's prior position was Advocate Fiscal of Ceylon.
  Fact 0.51: John Jeremie was from Guernsey, born 1795 and died 1841.
  Fact 0.52: John Jeremie's term was 10 March 1836 to 1840.
  Fact 0.53: John Jeremie was appointed by Wilmot-Horton.
  Fact 0.54: John Jeremie's prior position was public prosecutor of Mauritius.
  Fact 0.55: John Fredrick Stoddart was from Malta, born 1805 and died 1839.
  Fact 0.56: John Fredrick Stoddart's active service term was 2 June 1838 to 19 August 1839.
  Fact 0.57: John Fredrick Stoddart's length of active service was 443 days.
  Fact 0.58: John Fredrick Stoddart studied at the University of Glasgow.
  Fact 0.59: Anthony Oliphant was from Scotland, born 1793 and died 1859.
  Fact 0.60: Anthony Oliphant's active service term was 22 October 1838 to 1854.
  Fact 0.61: Anthony Oliphant's term as chief was 22 October 1838 to 1854.
  Fact 0.62: Anthony Oliphant's prior position was Attorney General of the Cape Colony.
  Fact 0.63: J.G. Hildebrand's term began 23 November 1839.
  Fact 0.64: William Ogle Carr was from England, born 1802 and died 1856.
  Fact 0.65: William Ogle Carr's term was 17 April 1854 to 1856.
  Fact 0.66: William Ogle Carr was appointed by Stewart-Mackenzie.
  Fact 0.67: William Ogle Carr's prior position was King's Advocate of Ceylon.
  Fact 0.68: John Stark's term began 10 October 1840.
  Fact 0.69: John Stark's prior position was Queen's Advocate of Ceylon.
  Fact 0.70: Christopher Temple's term began 28 January 1854.
  Fact 0.71: Christopher Temple's prior position was judge of the District Court of Colombo.
  Fact 0.72: Paul Ivy Sterling's term began 20 November 1855.
  Fact 0.73: Paul Ivy Sterling's prior position was Attorney General of Hong Kong.
  Fact 0.74: William Carpenter Rowe died in 1860.
  Fact 0.75: William Carpenter Rowe's term was 1857 to 1859.
  Fact 0.76: Edward Shepherd Creasy was from England, born 1812 and died 1878.
  Fact 0.77: Edward Shepherd Creasy's term was 27 March 1860 to 1875.
  Fact 0.78: Edward Shepherd Creasy studied at the University of Cambridge.
  Fact 0.79: Richard Morgan's prior position was Queen's Advocate of Ceylon.
  Fact 0.80: George Lawson has no recorded dates in the table.
  Fact 0.81: Henry Byerley Thompson died in 1867.
  Fact 0.82: Henry Byerley Thompson's term began 1 January 1863.
  Fact 0.83: Henry Byerley Thompson's prior position was Queen's Advocate of Ceylon.
  Fact 0.84: Charles Henry Stewart's term began 11 January 1867.
  Fact 0.85: Richard Cayley was from England.
  Fact 0.86: Richard Cayley's term was 26 July 1874 to 1882.
  Fact 0.87: Richard Cayley's term as chief was 1 October 1879 to 1882.
  Fact 0.88: Richard Cayley's prior position was Queen's Advocate of Ceylon.
  Fact 0.89: Lovell Burchett Clarence's term began 5 July 1879.
  Fact 0.90: Harry Dias Bandaranaike's term began 5 July 1879.
  Fact 0.91: William Hackett's term was 3 February 1877 to 1877.
  Fact 0.92: John Budd Phear was from England and his term began 18 October 1877.
  Fact 0.93: Archibald Campbell Lawrie lived from 1837 to 1914.
  Fact 0.94: Archibald Campbell Lawrie's term was 6 July 1892 to 1901.
  Fact 0.95: Archibald Campbell Lawrie's prior position was judge of the District Court of Kandy.
  Fact 0.96: Jacobus de Wet was from South Africa.
  Fact 0.97: Jacobus de Wet's term was 31 May 1882 to 29 May 1883, a length of 363 days.
  Fact 0.98: Bruce Burnside was from the Bahamas.
  Fact 0.99: Bruce Burnside's term was 21 May 1883 to 1893.
  Fact 0.100: Bruce Burnside's prior position was Queen's Advocate of Ceylon.
  Fact 0.101: John Winfield Bonser's term was 13 November 1893 to 1902.
  Fact 0.102: John Winfield Bonser's prior position was Chief Justice of the Straits Settlements.
  Fact 0.103: George Henry Withers has no recorded dates in the table.
  Fact 0.104: Dodwell Browne has no recorded dates in the table.
  Fact 0.105: Fredrick Charles Moncrieff's term began 4 April 1900.
  Fact 0.106: Henry Lorentz Wendt's term was 15 October 1901 to 28 November 1910.
  Fact 0.107: John Page Middleton's term was 28 January 1902 to 1912.
  Fact 0.108: John Page Middleton's prior position was judge of the Supreme Court of Cyprus.
  Fact 0.109: Charles Layard Western's term was 26 April 1902 to 18 June 1906, a length of 1,514 days.
  Fact 0.110: Joseph Richard Grenier's term was 28 November 1910 to 1912.
  Fact 0.111: Alexander Wood Renton's term was 3 September 1905 to 1918.
  Fact 0.112: Alexander Wood Renton's term as chief was 22 August 1914 to 1918.
  Fact 0.113: Alfred George Lascelles was from England and his term was 1 May 1911 to 1914.
  Fact 0.114: Joseph Turner Hutchinson was from England and his term was 23 October 1906 to 1 May 1911, a length of 1,651 days.
  Fact 0.115: James Cecil Walter Pereira's term began 4 July 1912.
  Fact 0.116: James Arthur Van Langenburg has no recorded dates in the table.
  Fact 0.117: Thomas De Sampayo's term began in 1915.
  Fact 0.118: Thomas De Sampayo studied at the University of Cambridge.
  Fact 0.119: Thomas De Sampayo's prior position was Commissioner of Assize.
  Fact 0.120: George Francis Macdaniel Ennis's term began 5 August 1912.
  Fact 0.121: Walter Shaw's term began 10 December 1914.
  Fact 0.122: Walter Shaw's prior position was Chief Justice of British Honduras.
  Fact 0.123: Gualterus Stewart Schneider has no recorded dates in the table.
  Fact 0.124: Anton Bertram lived from 1869 to 1937.
  Fact 0.125: Anton Bertram's term was 26 July 1918 to 1925.
  Fact 0.126: Anton Bertram studied at the University of Cambridge.
  Fact 0.127: Anton Bertram's prior position was Attorney General of Ceylon.
  Fact 0.128: Hermann Albert Loos has no recorded dates in the table.
  Fact 0.129: William Thomas Porter's term began 12 January 1922.
  Fact 0.130: Thomas Garvin lived from 1881 to 1940.
  Fact 0.131: Thomas Garvin's term was 21 June 1921 to 1935.
  Fact 0.132: Thomas Garvin studied at Colombo Law College.
  Fact 0.133: Thomas Garvin's prior position was Solicitor General of Ceylon.
  Fact 0.134: John Adrian St. Valentine Jayewardene lived from 1877 to 1927.
  Fact 0.135: John Adrian St. Valentine Jayewardene's term began in 1923.
  Fact 0.136: John Adrian St. Valentine Jayewardene was appointed by Manning.
  Fact 0.137: John Adrian St. Valentine Jayewardene's prior position was judge of the District Court of Colombo.
  Fact 0.138: Llwellyn Chisholm Dalton's term began 14 March 1925.
  Fact 0.139: Lewis Matthew Maartensz's term began 18 February 1935.
  Fact 0.140: Maas Thajoon Akbar lived from 1880 to 1944.
  Fact 0.141: Maas Thajoon Akbar studied at the University of Cambridge.
  Fact 0.142: Maas Thajoon Akbar's prior position was Solicitor General of Ceylon.
  Fact 0.143: Charles Ernest St. John Branch lived from 1865 to 1939.
  Fact 0.144: Charles Ernest St. John Branch's term was 3 July 1925 to 25 May 1926, a length of 326 days.
  Fact 0.145: Charles Ernest St. John Branch studied at Durham University.
  Fact 0.146: Stanley Fisher was born in 1867.
  Fact 0.147: Stanley Fisher's term was 11 December 1926 to 1930.
  Fact 0.148: Stanley Fisher's prior position was Chief Justice of Trinidad and Tobago.
  Fact 0.149: Robert William Lyall Grant's term began 17 February 1926.
  Fact 0.150: Allen Drieberg's term began 19 September 1927.
  Fact 0.151: Eugene Wilfred Jayewardene lived from 1874 to 1932.
  Fact 0.152: Eugene Wilfred Jayewardene's term began in May 1928.
  Fact 0.153: Eugene Wilfred Jayewardene's prior position was member of the Colombo Municipal Council.
  Fact 0.154: Philip James Macdonell lived from 1873 to 1940.
  Fact 0.155: Philip James Macdonell's term was 3 October 1930 to 1936.
  Fact 0.156: Philip James Macdonell's prior position was Chief Justice of Trinidad and Tobago.
  Fact 0.157: Lucian Macull Dominic de Silva has no recorded dates in the table.
  Fact 0.158: William Edward Barber has no recorded dates in the table.
  Fact 0.159: Francis Harold Bertram's term began 19 February 1935.
  Fact 0.160: Kenneth Poyser has no recorded dates in the table.
  Fact 0.161: Francis Soertsz's term began 12 November 1936.
  Fact 0.162: Sidney Abrahams was from England, born 1885 and died 1957.
  Fact 0.163: Sidney Abrahams's term was 3 July 1936 to 1939.
  Fact 0.164: Sidney Abrahams studied at the University of Cambridge.
  Fact 0.165: Francis Moseley's term began 2 August 1936.
  Fact 0.166: Vincent Fernando's term began 11 November 1937.
  Fact 0.167: Hector Hearne's term began 21 May 1937.
  Fact 0.168: Arthur Eric Keuneman's term began 4 March 1938.
  Fact 0.169: Arthur Eric Keuneman studied at the University of Cambridge.
  Fact 0.170: Oswald Leslie De Kretser II lived from 1882 to 1959.
  Fact 0.171: Oswald Leslie De Kretser II's term was 8 April 1938 to 1945.
  Fact 0.172: Oswald Leslie De Kretser II studied at Caldecott and University College, Colombo.
  Fact 0.173: Oswald Leslie De Kretser II's prior position was Commissioner of Assize.
  Fact 0.174: Arthur Wijewardena lived from 1887 to 1964.
  Fact 0.175: Arthur Wijewardena's term was 22 August 1938 to 1950.
  Fact 0.176: Arthur Wijewardena's term as chief was 15 January 1949 to 1950.
  Fact 0.177: Arthur Wijewardena studied at the University of Cambridge.
  Fact 0.178: Arthur Wijewardena's prior position was Solicitor General of Ceylon.
  Fact 0.179: J.H.B. Nihill's term began 18 November 1938.
  Fact 0.180: John Curtois Howard's term was 1 December 1939 to 1945.
  Fact 0.181: John Curtois Howard's prior position was Attorney General of Cyprus.
  Fact 0.182: G.H.F. Cannon has no recorded dates in the table.
  Fact 0.183: Edward Jayetileke's term was 14 May 1942 to 11 October 1951, a length of 3,437 days.
  Fact 0.184: Edward Jayetileke studied at Caldecott.
  Fact 0.185: Edward Jayetileke's prior position was Solicitor General of Ceylon.
  Fact 0.186: Alan Rose's term was 11 January 1945 to 1956.
  Fact 0.187: A.R.H. Canakeratne lived from 1886 to 1960.
  Fact 0.188: A.R.H. Canakeratne's term began 28 March 1945.
  Fact 0.189: A.R.H. Canakeratne studied at Ceylon Law College and the University of London.
  Fact 0.190: A.R.H. Canakeratne's prior position was Commissioner of Assize.
  Fact 0.191: Manikku Wadumestri Hendrick de Silva has no recorded dates in the table.
  Fact 0.192: Manikku Wadumestri Hendrick de Silva's prior position was Attorney General of Ceylon.
  Fact 0.193: Felix Reginald Dias Bandaranaike I lived from 1861 to 1947.
  Fact 0.194: Felix Reginald Dias Bandaranaike I's term began 3 September 1946.
  Fact 0.195: Felix Reginald Dias Bandaranaike I studied at the University of Cambridge.
  Fact 0.196: Felix Reginald Dias Bandaranaike I's prior position was judge of the District Court of Colombo.
  Fact 0.197: Chellappah Nagalingam lived from 1893 to 1958.
  Fact 0.198: Chellappah Nagalingam's term was 22 July 1947 to 1958.
  Fact 0.199: Chellappah Nagalingam studied at Ceylon Law College.
  Fact 0.200: Chellappah Nagalingam's prior position was Attorney General of Ceylon.
  Fact 0.201: Ralph Windham's term began 23 July 1947.
  Fact 0.202: Hema Henry Basnayake was born in 1902 and his term began 23 October 1947.
  Fact 0.203: Hema Henry Basnayake's term ended 3 August 1964, a length of 6,129 days.
  Fact 0.204: Hema Henry Basnayake's term as chief was 1 January 1956 to 3 August 1964, a length of 3,137 days.
  Fact 0.205: Hema Henry Basnayake's prior position was Attorney General of Ceylon.
  Fact 0.206: Edward Frederick Noel Gratiaen's term began 22 March 1948.
  Fact 0.207: E.H.T. Gunasekera's term was 1 March 1949 to 25 January 1962, a length of 4,713 days.
  Fact 0.208: M.F.S. Pulle's term began 28 September 1949.
  Fact 0.209: V.L. St. C. Swan's term began 1 May 1950.
  Fact 0.210: H.A. de Silva's term began 30 November 1950.
  Fact 0.211: Kairshasp Choksy has no recorded dates in the table.
  Fact 0.212: Henry Weerasooriya's term was 10 July 1953 to 24 February 1964, a length of 3,881 days.
  Fact 0.213: K.D. de Silva's term began 10 July 1953.
  Fact 0.214: Miliani Sansoni's term was 10 May 1954 to 17 November 1966, a length of 4,574 days.
  Fact 0.215: Miliani Sansoni's term as chief was 3 August 1964 to 17 November 1966, a length of 836 days.
  Fact 0.216: Hugh Fernando lived from 1910 to 1976.
  Fact 0.217: Hugh Fernando's term was 16 May 1955 to 17 November 1973, a length of 6,760 days.
  Fact 0.218: Hugh Fernando's term as chief was 20 November 1966 to 17 November 1973, a length of 2,554 days.
  Fact 0.219: Thusew Samuel Fernando's term began 2 May 1956.
  Fact 0.220: Thusew Samuel Fernando studied at Colombo Law College.
  Fact 0.221: Thusew Samuel Fernando's prior position was Solicitor General of Sri Lanka.
  Fact 0.222: Nadaraja Sinnetamby's term began 3 May 1956.
  Fact 0.223: Lionel de Silva's term began 5 November 1959.
  Fact 0.224: Leonard de Silva's term began 16 August 1966.
  Fact 0.225: Henry Thambiah was from the Northern province, born 1906 and died 1997.
  Fact 0.226: Henry Thambiah's term was 1 December 1961 to 27 May 1968, a length of 2,369 days.
  Fact 0.227: Henry Thambiah studied at University College, Colombo, Ceylon Law College and the University of London.
  Fact 0.228: Henry Thambiah's prior position was Commissioner of Assize.
  Fact 0.229: Kingsley Herath's term began 25 January 1962.
  Fact 0.230: Asoka Abeyesundere was from the Southern province, born 1906 and died 1988.
  Fact 0.231: Asoka Abeyesundere's term was 20 April 1962 to 3 May 1968, a length of 2,205 days.
  Fact 0.232: Asoka Abeyesundere studied at Ceylon Law College.
  Fact 0.233: Ponnuduraiswamy Sriskandarajah lived from 1905 to 1968.
  Fact 0.234: Ponnuduraiswamy Sriskandarajah's term was 22 June 1962 to 8 February 1967, a length of 1,692 days.
  Fact 0.235: Ponnuduraiswamy Sriskandarajah studied at Ceylon University College and Ceylon Law College.
  Fact 0.236: Ponnuduraiswamy Sriskandarajah's prior position was Commissioner of Assize.
  Fact 0.237: Gardiye Punchihewage Amaraseela Silva's term was 22 June 1962 to 1974.
  Fact 0.238: Gardiye Punchihewage Amaraseela Silva's term as chief was 1973 to 1974.
  Fact 0.239: Lionel Sirimanne has no recorded dates in the table.
  Fact 0.240: Vaithalingam Manicavasagar was from the Western province, born 1906 and died 1993.
  Fact 0.241: Vaithalingam Manicavasagar's term was 17 December 1964 to 14 December 1967, a length of 1,092 days.
  Fact 0.242: Vaithalingam Manicavasagar studied at Colombo Law College.
  Fact 0.243: Vaithalingam Manicavasagar's prior position was Commissioner of Assize.
  Fact 0.244: Anthony Alles was from the Central province, born 1911 and died 2003.
  Fact 0.245: Anthony Alles's term was 1964 to 1974.
  Fact 0.246: Anthony Alles studied at the University of Ceylon, Ceylon Law College and the University of London.
  Fact 0.247: Anthony Alles's prior position was Solicitor General of Sri Lanka.
  Fact 0.248: Veeravagu Sivasubramaniam lived from 1908 to 1985.
  Fact 0.249: Veeravagu Sivasubramaniam's term was 1966 to 1970.
  Fact 0.250: Veeravagu Sivasubramaniam studied at the University of London.
  Fact 0.251: Veeravagu Sivasubramaniam's prior position was Commissioner of Assize.
  Fact 0.252: George Samerawickrame's term was 18 November 1966 to 9 April 1981, a length of 5,256 days.
  Fact 0.253: Victor Tennekoon was from the Central province and was born in 1914.
  Fact 0.254: Victor Tennekoon's term as chief was 1 January 1974 to 8 September 1977.
  Fact 0.255: Victor Tennekoon studied at the University of London and Ceylon Law College.
  Fact 0.256: Victor Tennekoon's prior position was Judge of the Court of Appeal.
  Fact 0.257: Christopher Weeramantry was from the Western province, born 1926 and died 1967.
  Fact 0.258: Christopher Weeramantry's term began 30 April 1972.
  Fact 0.259: Christopher Weeramantry studied at the University of Ceylon and the University of London.
  Fact 0.260: Christopher Weeramantry's prior position was Commissioner of Assize.
  Fact 0.261: Oswald Leslie De Kretser III was from the Western province, born 1910 and died 1968.
  Fact 0.262: Oswald Leslie De Kretser III's term began 21 January 1972.
  Fact 0.263: Oswald Leslie De Kretser III studied at University College, Colombo and the University of London.
  Fact 0.264: Oswald Leslie De Kretser III's prior position was judge of the District Court of Panadura.
  Fact 0.265: Samarapilimudalige Wijayatilake was from the Central province, born 1912 and died 1990.
  Fact 0.266: Samarapilimudalige Wijayatilake's term was 1968 to 16 May 1975.
  Fact 0.267: Samarapilimudalige Wijayatilake studied at University College, Colombo and Ceylon Law College.
  Fact 0.268: Samarapilimudalige Wijayatilake's prior position was Commissioner of Assize.
  Fact 0.269: Pandita Gunawadena's term was 1968 to 10 February 1971.
  Fact 0.270: Vincent Thamotheram was born in 1915 and died on 23 March 1970.
  Fact 0.271: Vincent Thamotheram studied at Ceylon University College and Ceylon Law College.
  Fact 0.272: Vincent Thamotheram's prior position was Commissioner of Assize.
  Fact 0.273: Hector Deheragoda's term was 22 February 1972 to 27 December 1976, a length of 1,770 days.
  Fact 0.274: Cuda Banda Walgampaya's term was 22 February 1972 to 3 January 1976, a length of 1,411 days.
  Fact 0.275: Jaya Pathirana died in 2000.
  Fact 0.276: Jaya Pathirana's term was 22 February 1972 to 1978.
  Fact 0.277: Dharmadasa Wimalaratne's term began 10 April 1972.
  Fact 0.278: Dharmadasa Wimalaratne's term as chief was 20 October 1980 to 15 September 1985, a length of 1,791 days.
  Fact 0.279: Tellipalai Rajaratnam lived from 1920 to 1994.
  Fact 0.280: Tellipalai Rajaratnam's term was 10 April 1972 to 1978.
  Fact 0.281: Tellipalai Rajaratnam's prior position was Commissioner of Assize.
  Fact 0.282: Donald Sirimanne's term began in 1973.
  Fact 0.283: Cyril Udalagama's term began 1 January 1974.
  Fact 0.284: T.A. de S. Wijesundera's term began 1 January 1974.
  Fact 0.285: Malcolm Perera's term began 1 January 1974.
  Fact 0.286: Izadean Mohamed Ismail's term was 1 January 1974 to 18 December 1981, a length of 2,908 days.
  Fact 0.287: Joe Weereratne's term began 1 January 1974.
  Fact 0.288: Arulampalam Vythialingam's term began 1 January 1974.
  Fact 0.289: Noel Tittawela's term began 1 January 1974.
  Fact 0.290: Suppiah Sharvananda was from the Northern province, born 1923 and died 2007.
  Fact 0.291: Suppiah Sharvananda's term was 1 January 1974 to 1988.
  Fact 0.292: Suppiah Sharvananda's term as chief was 29 October 1984 to 1988.
  Fact 0.293: Suppiah Sharvananda studied at Colombo Law College and the University of London.
  Fact 0.294: Somaratne Walpita's term began 1 January 1974.
  Fact 0.295: Wilmot Gunasekera's term began 2 May 1974.
  Fact 0.296: Barnes Ratwatte II lived from 1918 to 2004.
  Fact 0.297: Barnes Ratwatte II's term began 2 December 1974.
  Fact 0.298: Barnes Ratwatte II's term as chief was 11 April 1981 to 22 November 1983, a length of 955 days.
  Fact 0.299: Barnes Ratwatte II studied at Ceylon Law College.
  Fact 0.300: Barnes Ratwatte II's prior position was President of the Court of Appeal of Sri Lanka.
  Fact 0.301: Raja Wanasundera was from the Sabaragamuwa province, born 1924 and died 2012.
  Fact 0.302: Raja Wanasundera's term began 5 June 1975.
  Fact 0.303: Raja Wanasundera studied at University College, Colombo, Ceylon Law College, the University of London and Stanford University.
  Fact 0.304: Raja Wanasundera's prior position was Solicitor General of Sri Lanka.
  Fact 0.305: Percy Colin-Thomé was from the Southern province, born 1922.
  Fact 0.306: Percy Colin-Thomé's term began 12 January 1976 and ran to 1977.
  Fact 0.307: Percy Colin-Thomé's term as chief began 29 December 1981.
  Fact 0.308: Percy Colin-Thomé studied at the University of Ceylon and the University of Cambridge.
  Fact 0.309: Percy Colin-Thomé's prior position was judge of the High Court of Colombo.
  Fact 0.310: Neville Samarakoon lived from 1919 to 1990.
  Fact 0.311: Neville Samarakoon's term as chief was 1977 to 21 October 1984.
  Fact 0.312: Neville Samarakoon studied at University College, Colombo and Colombo Law College.
  Fact 0.313: Victor Perera's term was 24 July 1981 to 18 August 1983, a length of 755 days.
  Fact 0.314: Anton Soza was from the Western province, born 1919 and died 2013.
  Fact 0.315: Anton Soza's term was 26 January 1982 to 3 February 1984, a length of 738 days.
  Fact 0.316: Anton Soza studied at the University of London.
  Fact 0.317: Anton Soza's prior position was Judge of the Court of Appeal.
  Fact 0.318: Parinda Ranasinghe's term was 15 November 1982 to 1991.
  Fact 0.319: Parinda Ranasinghe's term as chief was 1988 to 1991.
  Fact 0.320: Parinda Ranasinghe studied at Colombo Law College.
  Fact 0.321: Abdul Cader's term was 8 December 1982 to 28 May 1985, a length of 902 days.
  Fact 0.322: Heevi Rodrigo's term was 8 December 1982 to 4 December 1984, a length of 727 days.
  Fact 0.323: E.A.D. Atukorale's term began 27 November 1984.
  Fact 0.324: Herbert Thambiah was from the Northern province, born 1926 and died 1992.
  Fact 0.325: Herbert Thambiah's term began 27 November 1984 and ended in 1992.
  Fact 0.326: Herbert Thambiah's term as chief was in 1991.
  Fact 0.327: Herbert Thambiah studied at the University of Ceylon and Ceylon Law College.
  Fact 0.328: Herbert Thambiah's prior position was Judge of the Court of Appeal.
  Fact 0.329: Lucian Hector de Alwis's term began 27 November 1984.
  Fact 0.330: O.S.M. Seneviratne's term began 10 February 1986.
  Fact 0.331: H.A.G. de Silva's term began 10 February 1986.
  Fact 0.332: Mark Fernando lived from 1941 to 2009.
  Fact 0.333: Mark Fernando's term began 4 March 1988 and ended in 2005.
  Fact 0.334: Mark Fernando studied at the University of Ceylon, Peradeniya.
  Fact 0.335: A.R.B. Amerasinghe lived from 1937 to 2013.
  Fact 0.336: A.R.N. Fernando has no recorded dates in the table.
  Fact 0.337: Ameer Ismail's term was 1990 to 1994.
  Fact 0.338: Ameer Ismail studied at the University of Ceylon and Ceylon Law College.
  Fact 0.339: Ameer Ismail's prior position was President of the Court of Appeal of Sri Lanka.
  Fact 0.340: Andrew Ranjan Perera has no recorded dates in the table.
  Fact 0.341: Pathmanathan Ramanathan lived from 1932 to 2006.
  Fact 0.342: Pathmanathan Ramanathan studied at the University of Wales.
  Fact 0.343: Pathmanathan Ramanathan's prior position was President of the Court of Appeal of Sri Lanka.
  Fact 0.344: K.M.M.B. Kulatunge has no recorded dates in the table.
  Fact 0.345: G.P.S. de Silva's term was 1991 to 1999.
  Fact 0.346: Sarath N. Silva's term began 16 September 1999 and ended 7 June 2009, a length of 3,552 days.
  Fact 0.347: Sarath N. Silva was appointed by Kumaratunga.
  Fact 0.348: Sarath N. Silva studied at Sri Lanka Law College and the University of Brussels.
  Fact 0.349: Sarath N. Silva's prior position was President of the Court of Appeal of Sri Lanka.
  Fact 0.350: Shirani Bandaranayake was from the North Western province and was born in 1958.
  Fact 0.351: Shirani Bandaranayake was appointed by Kumaratunga.
  Fact 0.352: Shirani Bandaranayake studied at the University of Colombo and the University of London.
  Fact 0.353: Shirani Bandaranayake's prior position was lawyer.
  Fact 0.354: C.V. Vigneswaran was from the Western province, born 1939 and died in March 2001.
  Fact 0.355: C.V. Vigneswaran's term began in October 2004.
  Fact 0.356: C.V. Vigneswaran was appointed by Kumaratunga.
  Fact 0.357: C.V. Vigneswaran studied at the University of Ceylon and Ceylon Law College.
  Fact 0.358: C.V. Vigneswaran's prior position was Judge of the Court of Appeal.
  Fact 0.359: Asoka de Silva lived from 1946 to 1 August 2001.
  Fact 0.360: Asoka de Silva's term began 17 May 2011.
  Fact 0.361: Asoka de Silva's active service length was 3,576 days.
  Fact 0.362: Asoka de Silva's term as chief was 8 June 2009 to 17 May 2011, a length of 708 days.
  Fact 0.363: Asoka de Silva was appointed by Kumaratunga.
  Fact 0.364: Asoka de Silva studied at the University of Peradeniya, the University of Colombo and the University of Illinois at Urbana-Champaign.
  Fact 0.365: Asoka de Silva's prior position was President of the Court of Appeal of Sri Lanka.
  Fact 0.366: Nihal Jayasinghe's term was 19 February 2003 to 29 February 2008, a length of 1,836 days.
  Fact 0.367: Nihal Jayasinghe was appointed by Kumaratunga.
  Fact 0.368: Nihal Jayasinghe studied at the University of Colombo.
  Fact 0.369: Nihal Jayasinghe's prior position was Judge of the Court of Appeal.
  Fact 0.370: Nimal Dissanayake's term began 23 April 2008.
  Fact 0.371: Nimal Dissanayake was appointed by Kumaratunga.
  Fact 0.372: Shiranee Tilakawardane's term was 19 February 2003 to 5 April 2014, a length of 1,901 days.
  Fact 0.373: Shiranee Tilakawardane was appointed by Kumaratunga.
  Fact 0.374: Shiranee Tilakawardane studied at Sri Lanka Law College.
  Fact 0.375: Shiranee Tilakawardane's prior position was President of the Court of Appeal of Sri Lanka.
  Fact 0.376: Nimal Gamini Amaratunga's term began in January 2005 and ended in March 2014.
  Fact 0.377: Nimal Gamini Amaratunga was appointed by Kumaratunga.
  Fact 0.378: Nimal Gamini Amaratunga studied at the University of Colombo and the International Maritime Law Institute.
  Fact 0.379: Nimal Gamini Amaratunga's prior position was Judge of the Court of Appeal.
  Fact 0.380: Saleem Marsoof's term began in February 2005 and ended in 2014.
  Fact 0.381: Saleem Marsoof was appointed by Kumaratunga.
  Fact 0.382: Saleem Marsoof studied at the University of Colombo and the University of San Diego School of Law.
  Fact 0.383: Saleem Marsoof's prior position was President of the Court of Appeal of Sri Lanka.
  Fact 0.384: Andrew Somawansa was from the North Western province.
  Fact 0.385: Andrew Somawansa's term began 28 May 2006 and ended in October 2008.
  Fact 0.386: Andrew Somawansa was appointed by Rajapaksa.
  Fact 0.387: Andrew Somawansa studied at the University of Ceylon and the University of Nevada.
  Fact 0.388: Andrew Somawansa's prior position was President of the Court of Appeal of Sri Lanka.
  Fact 0.389: Jagath Balapatabendi was from the Southern province.
  Fact 0.390: Jagath Balapatabendi was appointed by Rajapaksa.
  Fact 0.391: Jagath Balapatabendi studied at Ceylon Law College.
  Fact 0.392: Jagath Balapatabendi's prior position was Judge of the Court of Appeal.
  Fact 0.393: Kanagasabapathy Sripavan was born in 1952.
  Fact 0.394: Kanagasabapathy Sripavan's term began 27 March 2008.
  Fact 0.395: Kanagasabapathy Sripavan's term as chief was 30 January 2015 to 28 February 2017.
  Fact 0.396: Kanagasabapathy Sripavan was appointed by Rajapaksa.
  Fact 0.397: Kanagasabapathy Sripavan studied at Sri Lanka Law College.
  Fact 0.398: Kanagasabapathy Sripavan's prior position was Judge of the Court of Appeal.
  Fact 0.399: P.A. Ratnayake's term was 2 May 2008 to 15 February 2013, a length of 1,751 days.
  Fact 0.400: P.A. Ratnayake was appointed by Rajapaksa.
  Fact 0.401: P.A. Ratnayake studied at the University of Brussels.
  Fact 0.402: Chandra Ekanayake's term was November 2008 to 2016.
  Fact 0.403: Chandra Ekanayake was appointed by Rajapaksa.
  Fact 0.404: Chandra Ekanayake studied at Sri Lanka Law College.
  Fact 0.405: Chandra Ekanayake's prior position was President of the Court of Appeal of Sri Lanka.
  Fact 0.406: S.I. Imam was from the Central province.
  Fact 0.407: S.I. Imam's term was 9 January 2009 to 20 February 2013, a length of 1,503 days.
  Fact 0.408: S.I. Imam was appointed by Rajapaksa.
  Fact 0.409: S.I. Imam's prior position was Judge of the Court of Appeal.
  Fact 0.410: R.K.S. Suresh Chandra's term began 17 June 2010.
  Fact 0.411: R.K.S. Suresh Chandra was appointed by Rajapaksa.
  Fact 0.412: Sathya Hettige's term was 10 July 2011 to 18 April 2014, a length of 1,013 days.
  Fact 0.413: Sathya Hettige was appointed by Rajapaksa.
  Fact 0.414: Sathya Hettige studied at the University of Colombo.
  Fact 0.415: Sathya Hettige's prior position was President of the Court of Appeal of Sri Lanka.
  Fact 0.416: Priyasath Dep's term began 10 July 2011 and is ongoing.
  Fact 0.417: Priyasath Dep's term as chief began 2 March 2017.
  Fact 0.418: Priyasath Dep was appointed by Rajapaksa.
  Fact 0.419: Priyasath Dep studied at the University of Colombo and the International Institute of Social Studies.
  Fact 0.420: Priyasath Dep's prior position was Solicitor General of Sri Lanka.
  Fact 0.421: Eva Wansundera was from the North Western province.
  Fact 0.422: Eva Wansundera's term began 7 July 2012 and is ongoing.
  Fact 0.423: Eva Wansundera was appointed by Rajapaksa.
  Fact 0.424: Eva Wansundera studied at Sri Lanka Law College and the University of Leicester.
  Fact 0.425: Eva Wansundera's prior position was Attorney General of Sri Lanka.
  Fact 0.426: Mohan Peiris's term was 15 January 2013 to 28 January 2015, a length of 743 days.
  Fact 0.427: Mohan Peiris was appointed by Rajapaksa.
  Fact 0.428: Mohan Peiris studied at Sri Lanka Law College.
  Fact 0.429: Mohan Peiris's prior position was Attorney General of Sri Lanka.
  Fact 0.430: Rohini Marasinghe's term was 26 April 2013 to 2015.
  Fact 0.431: Rohini Marasinghe was appointed by Rajapaksa.
  Fact 0.432: Rohini Marasinghe studied at the University of Hartford.
  Fact 0.433: Rohini Marasinghe's prior position was Judge of the Court of Appeal.
  Fact 0.434: Buwaneka Aluwihare's term began 4 December 2013 and is ongoing.
  Fact 0.435: Buwaneka Aluwihare was appointed by Rajapaksa.
  Fact 0.436: Buwaneka Aluwihare studied at the University of London.
  Fact 0.437: Buwaneka Aluwihare's prior position was Additional Solicitor General of Sri Lanka.
  Fact 0.438: Sisira de Abrew was from the Western province.
  Fact 0.439: Sisira de Abrew's term began 7 May 2014 and is ongoing.
  Fact 0.440: Sisira de Abrew was appointed by Rajapaksa.
  Fact 0.441: Sisira de Abrew studied at Sri Lanka Law College.
  Fact 0.442: Sisira de Abrew's prior position was President of the Court of Appeal of Sri Lanka.
  Fact 0.443: Sarath de Abrew lived from 1953 to 2016.
  Fact 0.444: Sarath de Abrew's term began 7 May 2014 and ended in January 2016.
  Fact 0.445: Sarath de Abrew was appointed by Rajapaksa.
  Fact 0.446: Sarath de Abrew studied at Sri Lanka Law College.
  Fact 0.447: Sarath de Abrew's prior position was Judge of the Court of Appeal.
  Fact 0.448: Priyantha Jayawardena's term began 7 May 2014 and is ongoing.
  Fact 0.449: Priyantha Jayawardena was appointed by Rajapaksa.
  Fact 0.450: Priyantha Jayawardena studied at Sri Lanka Law College and the University of Aberdeen.
  Fact 0.451: Upali Abeyratne's term began 17 December 2014 and is ongoing.
  Fact 0.452: Upali Abeyratne was appointed by Rajapaksa.
  Fact 0.453: Upali Abeyratne's prior position was Judge of the Court of Appeal.
  Fact 0.454: Anil Goonaratne's term began 30 January 2015 and is ongoing.
  Fact 0.455: Anil Goonaratne was appointed by Sirisena.
  Fact 0.456: Anil Goonaratne studied at Sri Lanka Law College and the University of Exeter.
  Fact 0.457: Anil Goonaratne's prior position was Judge of the Court of Appeal.
  Fact 0.458: Kankanithanthri Chitrasiri's term began 3 December 2015 and is ongoing.
  Fact 0.459: Kankanithanthri Chitrasiri was appointed by Sirisena.
  Fact 0.460: Kankanithanthri Chitrasiri studied at the University of Sri Lanka and Queen Mary University of London.
  Fact 0.461: Kankanithanthri Chitrasiri's prior position was Judge of the Court of Appeal.
  Fact 0.462: Nalin Perera's term began 3 March 2016 and is ongoing.
  Fact 0.463: Nalin Perera was appointed by Sirisena.
  Fact 0.464: Nalin Perera's prior position was Judge of the Court of Appeal.
  Fact 0.465: Prasanna Sujeewa Jayawardena's term began 16 June 2016 and is ongoing.
  Fact 0.466: Prasanna Sujeewa Jayawardena was appointed by Sirisena.
  Fact 0.467: Prasanna Sujeewa Jayawardena studied at the University of Colombo.
  Fact 0.468: Prasanna Sujeewa Jayawardena's prior position was Judge of the Court of Appeal.

Retrieved items:
<item 1>
List of Justices of the Supreme Court of Sri Lanka
List of justices of the Supreme Court of Sri Lanka - wikipedia List of justices of the Supreme Court of Sri Lanka Jump to : navigation , search Shirani
Bandaranayake was the 43rd Chief Justice of Sri Lanka and the first and only woman to hold the position . List of Justices of the Supreme Court of Sri Lanka In
this list , justices are placed in the order in which they took the judicial oath of office and thereby started their term of office . While many of the
justices ' positions prior to appointment are simply listed as `` lawyer '' , many had part - time positions , such as teaching , or acted as counsel to various
levels of government . Contents ( hide ) 1 Justices 2 See also 3 Notes 4 References 5 Bibliography Justices ( edit ) † Denotes current justices Began as
Provisional or Acting Puisne Justice ♠ Tenure demoted to de facto † Died in office No .
</item 1>

<item 2>
List of Justices of the Supreme Court of Sri Lanka
182 -- Sirisena Sri Lanka Law College University of Exeter Judge of the Court of Appeal Kankanithanthri Chitrasiri † 3 December 2015 -- present --
7002875000000000000 ♠ 875 -- Sirisena University of Sri Lanka Queen Mary University of London Judge of the Court of Appeal Nalin Perera † 3 March 2016 --
present -- 7002784000000000000 ♠ 784 -- Sirisena Judge of the Court of Appeal Prasanna Sujeewa Jayawardena † 16 June 2016 -- present -- 7002679000000000000 ♠
679 -- Sirisena University of Colombo Judge of the Court of Appeal See also ( edit ) List of Chief Justices of Sri Lanka List of Justices of the Supreme Court
of Sri Lanka by court composition Notes ( edit ) Jump up ^ Provisional Chief Justice Jump up ^ On leave ( 1 January - 28 April 1871 ) Jump up ^ Acting Chief
Justice ( 1 - 10 October 1879 ,
</item 2>

<item 3>
List of Justices of the Supreme Court of Sri Lanka
Jump up ^ Current to most recent page update . Bibliography ( edit ) Amerasinghe , A.R.B. ( 1986 ) . The Supreme Court of Sri Lanka : the first 185 years .
Ratmalana , Sri Lanka : Sarvodaya Book Pub . Services . ISBN 955599000X . ( hide ) Supreme Court of Sri Lanka The Court Decisions Procedure History Court
Building Current membership Chief Justice Kanagasabapathy Sripavan Puisne Justices Priyasath Dep Eva Wansundera Buwaneka Aluwihare Sisira De Abrew Priyantha
Jayawardena Upaly Abeyratne Anil Goonaratne K.T.
</item 3>

<item 4>
List of Justices of the Supreme Court of Sri Lanka
20 October 1980 -- 15 September 1985 -- 1,791 -- Bandaranaike 122 Tellipalai Rajaratnam 1920 - 1994 10 April 1972 -- 1978 -- -- Bandaranaike Commissioner of
Assize 123 Donald Sirimanne 1973 -- -- -- Gopallawa 124 Cyril Udalagama 1 January 1974 -- -- -- Gopallawa 125 T.A. de S. Wijesundera 1 January 1974 -- -- --
Gopallawa 126 Malcolm Perera 1 January 1974 -- -- -- Gopallawa 127 Izadean Mohamed Ismail 1 January 1974 -- 18 December 1981 -- 2,
</item 4>

<item 5>
List of Justices of the Supreme Court of Sri Lanka
Imam Central 9 January 2009 -- 20 February 2013 -- 1,503 -- Rajapaksa Judge of the Court of Appeal R.K.S. Suresh Chandra 17 June 2010 -- -- -- Rajapaksa Sathya
Hettige 10 July 2011 -- 18 April 2014 -- 1,013 -- Rajapaksa University of Colombo President of the Court of Appeal of Sri Lanka Priyasath Dep † 10 July 2011 --
present 02 March 2017 - present -- 7003248200000000000 ♠ 2,482 -- Rajapaksa University of Colombo International Institute of Social Studies Solicitor General of
Sri Lanka Eva Wansundera † North Western 7 July 2012 -- present -- 7003211900000000000 ♠ 2,
</item 5>

<item 6>
List of Justices of the Supreme Court of Sri Lanka
760 2,554 Kotelawala 98 Thusew Samuel Fernando 2 May 1956 -- -- -- Bandaranaike Colombo Law College Solicitor General of Sri Lanka 99 Nadaraja Sinnetamby 3 May
1956 -- -- -- Bandaranaike 100 Lionel de Silva 5 November 1959 -- -- -- Dahanayake 101 Leonard de Silva 16 August 1966 -- -- -- Senanayake 102 Henry Thambiah
Northern 1906 -- 1997 1 December 1961 -- 27 May 1968 -- 2,369 -- Bandaranaike University College , Colombo Ceylon Law College University of London Commissioner
of Assize 103 Kingsley Herath 25 January 1962 -- -- -- Bandaranaike 104 Asoka Abeyesundere Southern 1906 - 1988 20 April 1962 -- 3 May 1968 -- 2,
</item 6>

<item 7>
List of Justices of the Supreme Court of Sri Lanka
Kulatunge -- -- -- G.P.S. de Silva -- 1991 -- 1999 Sarath N. Silva 1995 -- 16 September 1999 16 September 1999 -- 7 June 2009 3,552 Kumaratunga Sri Lanka Law
College University of Brussels President of the Court of Appeal of Sri Lanka Shirani Bandaranayake North Western 1958 - 30 October 1996 -- 13 January 2013 28
January 2015 -- 29 January 2015 18 May 2011 -- 13 January 2013 28 January 2015 -- 29 January 2015 5,919 + 1 606 + 1 Kumaratunga University of Colombo University
of London Lawyer C.V. Vigneswaran Western 1939 - March 2001 -- October 2004 -- -- Kumaratunga University of Ceylon Ceylon Law College Judge of the Court of
Appeal Asoka de Silva 1946 - 1 August 2001 -- 17 May 2011 8 June 2009 -- 17 May 2011 3,
</item 7>

<item 8>
List of Justices of the Supreme Court of Sri Lanka
450 -- Rajapaksa Sri Lanka Law College President of the Court of Appeal of Sri Lanka Sarath de Abrew 1953 -- 2016 7 May 2014 -- January 2016 -- -- Rajapaksa Sri
Lanka Law College Judge of the Court of Appeal Priyantha Jayawardena † 7 May 2014 -- present -- 7003145000000000000 ♠ 1,450 -- Rajapaksa Sri Lanka Law College
University of Aberdeen Upali Abeyratne † 17 December 2014 -- present -- 7003122600000000000 ♠ 1,226 -- Rajapaksa Judge of the Court of Appeal Anil Goonaratne †
30 January 2015 -- present -- 7003118200000000000 ♠ 1,
</item 8>

<item 9>
List of Justices of the Supreme Court of Sri Lanka
University of San Diego School of Law President of the Court of Appeal of Sri Lanka Andrew Somawansa North Western 28 May 2006 -- October 2008 -- -- Rajapaksa
University of Ceylon University of Nevada President of the Court of Appeal of Sri Lanka Jagath Balapatabendi Southern -- -- -- Rajapaksa Ceylon Law College
Judge of the Court of Appeal Kanagasabapathy Sripavan † 1952 -- 27 March 2008 -- present 30 January 2015 -- 28 February 2017 7003368200000000000 ♠ 3,682
7003118200000000000 ♠ 1,182 Rajapaksa Sri Lanka Law College Judge of the Court of Appeal P.A. Ratnayake 2 May 2008 -- 15 February 2013 -- 7003175100000000000 ♠
1,751 -- Rajapaksa University of Brussels Chandra Ekanayake November 2008 -- 2016 -- -- Rajapaksa Sri Lanka Law College President of the Court of Appeal of Sri
Lanka S.I.
</item 9>

<item 10>
List of Justices of the Supreme Court of Sri Lanka
9 January - 5 June 1888 ) Jump up ^ On leave ( 25 November 1890 - February 1891 ) Jump up ^ Acting Justice ( 12 May 1903 -- 28 November 1910 ) References ( edit
) Jump up ^ Amerasinghe , A. Ranjit B ( 1986 ) , The Supreme Court of Sri Lanka : the first 185 years , Sarvodaya Book Pub . Services , ISBN 978 - 955 - 599 -
000 - 4 Jump up ^ Current to most recent page update . Jump up ^ Current to most recent page update . Jump up ^ Current to most recent page update . Jump up ^
Current to most recent page update . Jump up ^ Current to most recent page update . Jump up ^ Current to most recent page update . Jump up ^ Current to most
recent page update . Jump up ^ Current to most recent page update . Jump up ^ Current to most recent page update . Jump up ^ Current to most recent page update
. Jump up ^ Current to most recent page update . Jump up ^ Current to most recent page update .
</item 10>
=== END ITEM id=b7572fbf0fba5aa1 ===

=== ITEM id=fd1372b634f14a4a ===
Gold sentences and facts:
Sentence 0: The gymnosperms are a group of seed - producing plants that includes conifers , cycads , Ginkgo , and gnetophytes . The term `` gymnosperm '' comes from the Greek composite word γυμνόσπερμος ( γυμνός gymnos , `` naked '' and σπέρμα sperma , `` seed '' ) , meaning `` naked seeds '' . The name is based on the unenclosed condition of their seeds ( called ovules in their unfertilized state ) . The non-encased condition of their seeds stands in contrast to the seeds and ovules of flowering plants ( angiosperms ) , which are enclosed within an ovary . Gymnosperm seeds develop either on the surface of scales or leaves , which are often modified to form cones , or solitary as in Yew , Torreya , Ginkgo .
  Fact 0.0: The gymnosperms are a group of seed-producing plants.
  Fact 0.1: Gymnosperms include conifers, cycads, Ginkgo, and gnetophytes.
  Fact 0.2: The term gymnosperm comes from the Greek composite word γυμνόσπερμος.
  Fact 0.3: The Greek word γυμνός (gymnos) means naked.
  Fact 0.4: The Greek word σπέρμα (sperma) means seed.
  Fact 0.5: The term gymnosperm means naked seeds.
  Fact 0.6: The name gymnosperm is based on the unenclosed condition of gymnosperm seeds.
  Fact 0.7: Gymnosperm ovules are called ovules in their unfertilized state.
  Fact 0.8: Gymnosperm seeds are not encased.
  Fact 0.9: The seeds and ovules of flowering plants (angiosperms) are enclosed within an ovary.
  Fact 0.10: Gymnosperm seeds develop either on the surface of scales or leaves.
  Fact 0.11: The scales or leaves on which gymnosperm seeds develop are often modified to form cones.
  Fact 0.12: Gymnosperm seeds can also be solitary, as in Yew, Torreya, and Ginkgo.

Retrieved items:
<item 1>
Gymnosperm
The name is based on the unenclosed condition of their seeds ( called ovules in their unfertilized state ) . The non-encased condition of their seeds stands in
contrast to the seeds and ovules of flowering plants ( angiosperms ) , which are enclosed within an ovary . Gymnosperm seeds develop either on the surface of
scales or leaves , which are often modified to form cones , or solitary as in Yew , Torreya , Ginkgo . The gymnosperms and angiosperms together compose the
spermatophytes or seed plants . The gymnosperms are divided into six phyla . Organisms that belong to the Cycadophyta , Ginkgophyta , Gnetophyta , and Pinophyta
( also known as Coniferophyta ) phyla are still in existence while those in the Pteridospermales and Cordaitales phyla are now extinct .
</item 1>

<item 2>
Gymnosperm
Bract Pedicellate Flower Whorl Floral symmetry Floral diagram Floral formula Receptacle Hypanthium ( Floral cup ) Perianth Tepal Petal Sepal Sporophyll
Gynoecium Ovary Ovule Stigma Archegonium Androecium Stamen Staminode Pollen Tapetum Gynandrium Gametophyte Sporophyte Plant embryo Fruit Fruit anatomy Berry
Capsule Seed Seed dispersal Endosperm Surface structures Epicuticular wax Plant cuticle Epidermis Stoma Nectary Trichome Prickle Plant physiology Materials
Nutrition Photosynthesis Chlorophyll Plant hormone Transpiration Turgor pressure Bulk flow Aleurone Phytomelanin Sugar Sap Starch Cellulose Plant growth and
habit Secondary growth Woody plants Herbaceous plants Habit Vines Lianas Shrubs Subshrubs Trees Succulent plants Reproduction Evolution Ecology Alternation of
generations Sporangium Spore Microsporangia Microspore Megasporangium Megaspore Pollination Pollinators Pollen tube Double fertilization Germination
Evolutionary development Evolutionary history timeline Hardiness zone Plant taxonomy History of plant systematics Herbarium Biological classification Botanical
nomenclature Botanical name Correct
</item 2>

<item 3>
Gymnosperm
Agathis in Araucariaceae and Nageia in Podocarpaceae have broad , flat strap - shaped leaves . Cycads are the next most abundant group of gymnosperms , with two
or three families , 11 genera , and approximately 338 species . A majority of cycads are native to tropical climates and are most abundantly found in regions
near the equator . The other extant groups are the 95 - 100 species of Gnetales and one species of Ginkgo . Uses ( edit ) Gymnosperms have major economic uses .
Pine , fir , spruce , and cedar are all examples of conifers that are used for lumber , paper production , and resin . Some other common uses for gymnosperms
are soap , varnish , nail polish , food , gum , and perfumes . Life cycle ( edit ) Example of gymnosperm lifecycle Gymnosperms , like all vascular plants , have
a sporophyte - dominant life cycle , which means they spend most of their life cycle with diploid cells , while the gametophyte ( gamete - bearing phase ) is
relatively short - lived .
</item 3>

<item 4>
Gymnosperm
After syngamy ( joining of the sperm and egg cell ) , the zygote develops into an embryo ( young sporophyte ) . More than one embryo is usually initiated in
each gymnosperm seed . The mature seed comprises the embryo and the remains of the female gametophyte , which serves as a food supply , and the seed coat .
Genetics ( edit ) The first published sequenced genome for any gymnosperm was the genome of Picea abies in 2013 . References ( edit ) ^ Jump up to : ``
Gymnosperms on The Plant List '' . Theplantlist.org . Retrieved 2013 - 07 - 24 . ^ Jump up to : Raven , P.H. ( 2013 ) . Biology of Plants . New York : W.H.
Freeman and Co . Jump up ^ Jeffrey D. Palmer ; Douglas E. Soltis ; Mark W. Chase ( 2004 ) . `` The plant tree of life : an overview and some points of view '' .
American Journal of Botany. 91 ( 10 ) : 1437 -- 1445 .
</item 4>

<item 5>
Gymnosperm
Two spore types , microspores and megaspores , are typically produced in pollen cones or ovulate cones , respectively . Gametophytes , as with all heterosporous
plants , develop within the spore wall . Pollen grains ( microgametophytes ) mature from microspores , and ultimately produce sperm cells . Megagametophytes
develop from megaspores and are retained within the ovule . Gymnosperms produce multiple archegonia , which produce the female gamete . During pollination ,
pollen grains are physically transferred between plants from the pollen cone to the ovule . Pollen is usually moved by wind or insects . Whole grains enter each
ovule through a microscopic gap in the ovule coat ( integument ) called the micropyle . The pollen grains mature further inside the ovule and produce sperm
cells . Two main modes of fertilization are found in gymnosperms . Cycads and Ginkgo have motile sperm that swim directly to the egg inside the ovule , whereas
conifers and gnetophytes have sperm with no flagella that are moved along a pollen tube to the egg .
</item 5>

<item 6>
Gymnosperm
91 ( 10 ) : 1437 -- 1445 . doi : 10.3732 / ajb. 91.10. 1437 . PMID 21652302 . Jump up ^ Stevens , P.F. `` Angiosperm Phylogeny Website - Seed Plant Evolution ''
. Jump up ^ Hilton , Jason , and Richard M. Bateman. 2006 . Pteridosperms are the backbone of seed - plant phylogeny . Journal of the Torrey Botanical Society
133 : 119 - 168 ( abstract ) Jump up ^ Christenhusz , M.J.M. ; Reveal , J.L. ; Farjon , A. ; Gardner , M.F. ; Mill , R.R. ; Chase , M.W. ( 2011 ) . `` A new
classification and linear sequence of extant gymnosperms '' ( PDF ) . Phytotaxa. 19 : 55 -- 70 . Jump up ^ Christenhusz , M.J.M. ; Byng , J.W. ( 2016 ) .
</item 6>

<item 7>
Gymnosperm
External links ( edit ) Wikimedia Commons has media related to Gymnosperm . Gymnosperm Database Gymnosperms on the Tree of Life Albert Seward ( 1911 ) . ``
Gymnosperms '' . Encyclopædia Britannica ( 11th ed . ) . Botany History of botany Subdisciplines Plant systematics Ethnobotany Paleobotany Plant anatomy Plant
ecology Phytogeography Geobotany Flora Phytochemistry Plant pathology Bryology Phycology Floristics Dendrology Plant groups Algae Archaeplastida Bryophyte
Non-vascular plants Vascular plants Spermatophytes Pteridophyte Gymnosperm Angiosperm Plant morphology ( glossary ) Plant cells Cell wall Phragmoplast Plastid
Plasmodesma Vacuole Tissues Meristem Vascular tissue Vascular bundle Ground tissue Mesophyll Cork Wood Storage organs Vegetative Root Rhizoid Bulb Rhizome Shoot
Stem Leaf Petiole Cataphyll Bud Sessility Reproductive ( Flower ) Flower development Inflorescence Umbel Raceme Bract Pedicellate Flower Whorl Floral symmetry
Floral diagram Floral formula Receptacle
</item 7>

<item 8>
Gymnosperm
Diselma , Fitzroya , Callitris ( incl . Actinostrobus and Neocallitropsis ) , Thujopsis , Thuja , Fokienia , Chamaecyparis , Callitropsis , Cupressus ,
Juniperus , Xanthocyparis , Calocedrus , Tetraclinis , Platycladus , Microbiota Family Taxaceae : Austrotaxus , Pseudotaxus , Taxus , Cephalotaxus , Amentotaxus
, Torreya Diversity and origin ( edit ) There are more than 1000 extant or currently living species of gymnosperms in 88 plant genera belonging to 14 plant
families . It is widely accepted that the gymnosperms originated in the late Carboniferous period , replacing the lycopsid rainforests of the tropical region .
This appears to have been the result of a whole genome duplication event around 319 million years ago . Early characteristics of seed plants were evident in
fossil progymnosperms of the late Devonian period around 383 million years ago .
</item 8>

<item 9>
Gymnosperm
There is conflicting evidence on the question of whether the living gymnosperms form a clade . The fossil record of gymnosperms includes many distinctive taxa
that do not belong to the four modern groups , including seed - bearing trees that have a somewhat fern - like vegetative morphology ( the so - called `` seed
ferns '' or pteridosperms . ) When fossil gymnosperms such as Bennettitales , Caytonia and the glossopterids are considered , it is clear that angiosperms are
nested within a larger gymnosperm clade , although which group of gymnosperms is their closest relative remains unclear . For the most recent classification on
extant gymnosperms see Christenhusz et al. ( 2011 ) . There are 12 families , 83 known genera with a total of ca 1080 known species ( Christenhusz & Byng 2016 )
.
</item 9>

<item 10>
Gymnosperm
Gymnosperm - wikipedia Gymnosperm Jump to : navigation , search Gymnosperms Temporal range : 370 -- 0 Ma PreЄ Є Pg Devonian - Present Various gymnosperms .
Scientific classification Kingdom : Plantae Subkingdom : Embryophyta ( unranked ) : Gymnospermae ( paraphyletic ) Divisions Pinophyta ( or Coniferophyta ) -
Conifers Ginkgophyta - Ginkgo Cycadophyta - Cycads Gnetophyta - Gnetum , Ephedra , Welwitschia Encephalartos sclavoi cone , about 30 cm long The gymnosperms are
a group of seed - producing plants that includes conifers , cycads , Ginkgo , and gnetophytes . The term `` gymnosperm '' comes from the Greek composite word
γυμνόσπερμος ( γυμνός gymnos , `` naked '' and σπέρμα sperma , `` seed '' ) , meaning `` naked seeds '' .
</item 10>
=== END ITEM id=fd1372b634f14a4a ===

=== ITEM id=eaf0218af501256b ===
Gold sentences and facts:
Sentence 0: Ford was involved in the Boy Scouts of America , and earned that program 's highest rank , Eagle Scout . He is the only Eagle Scout to have ascended to the U.S. Presidency .
  Fact 0.0: Ford was involved in the Boy Scouts of America.
  Fact 0.1: Ford earned the Boy Scouts of America program's highest rank, Eagle Scout.
  Fact 0.2: Ford is the only Eagle Scout to have ascended to the U.S. Presidency.

Retrieved items:
<item 1>
Gerald Ford
Ford was made a 33 ° Scottish Rite Mason on September 26 , 1962 . In April 1975 , Ford was elected by a unanimous vote Honorary Grand Master of the
International Supreme Council , Order of DeMolay , a position in which he served until January 1977 . Ford received the degrees of York Rite Masonry ( Chapter
and Council degrees ) in a special ceremony in the Oval Office on January 11 , 1977 , during his term as President of the United States . Honors ( edit ) Ford
honored on U.S Postage , issue of 2007 Ford received the Distinguished Eagle Scout Award in May 1970 , as well as the Silver Buffalo Award , from the Boy Scouts
of America . In 1974 , he also received the highest distinction of the Scout Association of Japan , the Golden Pheasant Award . In 1985 , he received the 1985
Old Tom Morris Award from the Golf Course Superintendents Association of America , GCSAA 's highest honor . In 1992 , the U.S. Navy Memorial Foundation awarded
Ford its Lone Sailor Award for his naval service and his subsequent government service .
</item 1>

<item 2>
Gerald Ford served eight months as vice president after Spiro Agnew resigned, then became president without being elected to either executive office—the only
person to do so. A Michigan representative for 25 years, including nine as House minority leader, he signed the Helsinki Accords, oversaw the effective end of
U.S. involvement in Vietnam, and faced severe inflation and recession. His pardon of Richard Nixon for Watergate was highly controversial.
</item 2>

<item 3>
President of the United States
Boy Scouts of America . Retrieved November 9 , 2017 . Jump up ^ Grier , Peter ( April 25 , 2011 ) . `` The ( not so ) secret history of the White House Easter
Egg Roll '' . The Christian Science Monitor . Archived from the original on July 30 , 2012 . Retrieved July 30 , 2012 . Jump up ^ Hesse , Monica ( November 21 ,
2007 ) . `` Turkey Pardons , The Stuffing of Historic Legend '' . The Washington Post . Retrieved May 14 , 2011 . Jump up ^ Gibbs , Nancy ( November 13 , 2008 )
. `` How Presidents Pass The Torch '' . Time . Retrieved May 6 , 2011 . Jump up ^ Dorning , Mike ( January 22 , 2009 ) . `` A note from Bush starts morning in
the Oval Office '' . Chicago Tribune . Archived from the original on December 28 , 2011 . Retrieved May 6 , 2011 .
</item 3>

<item 4>
The president has ceremonial duties, including maintaining traditions begun by predecessors. William Howard Taft started the presidential first-pitch tradition
in 1910; nearly every president since has thrown a ceremonial pitch, and presidents have served as honorary presidents of the Boy Scouts of America. Rutherford
B. Hayes began the White House Easter egg roll in 1878. The text also notes that “POTUS” originated as telegraph shorthand in 1879 and lists nine vice
presidents who completed a predecessor’s term
</item 4>

<item 5>
Gerald Ford’s biological father briefly reconnected with him while Ford worked in a Grand Rapids restaurant; they stayed sporadically in touch until his
father’s death in 1941. Ford praised his stepfather and mother for providing a superb upbringing. An Eagle Scout—the only one to become U.S. president—Ford was
a star athlete and football captain at Grand Rapids South High, made the city’s All-City team in 1930, and later played center at the University of Michigan.
</item 5>

<item 6>
Gerald Ford
Legacy of Honor : The Values and Influence of America 's Eagle Scouts . New York : St. Martin 's Press . pp. 12 -- 13 and 87 . ISBN 0 - 312 - 36653 - 1 .
Retrieved December 29 , 2006 . Jump up ^ `` Investigatory Records on Gerald Ford , Applicant for a Commission '' ( PDF ) . Gerald R. Ford Presidential Library .
December 30 , 1941 . Retrieved November 18 , 2010 . Jump up ^ Wertheimer , Linda ( December 27 , 2006 ) . `` Special Report : Former President Gerald Ford Dies
; Sought to Heal Nation Disillusioned by Watergate Scandal '' . National Public Radio . Retrieved April 26 , 2009 . Jump up ^ Perry , Will ( 1974 ) . `` No
Cheers From the Alumni '' . The Wolverines : A Story of Michigan Football ( PDF ) . Huntsville , Alabama : The Strode Publishers . pp. 150 -- 152 . ISBN 0 -
87397 - 055 - 1 .
</item 6>

<item 7>
The references cover U.S. presidents’ ceremonial and transition traditions: the White House Easter Egg Roll (Peter Grier, 2011), turkey pardons (Monica Hesse,
2007), presidential transitions (Nancy Gibbs, 2008), and a note George W. Bush left for Barack Obama in the Oval Office (Mike Dorning, 2009). They also cite a
source on presidents and the Boy Scouts of America.
</item 7>

<item 8>
Gerald Ford
That year his biological father , whom Ford described as a `` carefree , well - to - do man who did n't really give a damn about the hopes and dreams of his
firstborn son '' , approached Ford while he was waiting tables in a Grand Rapids restaurant . The two `` maintained a sporadic contact '' until Leslie King ,
Sr. 's death in 1941 . Ford said , `` My stepfather was a magnificent person and my mother equally wonderful . So I could n't have written a better prescription
for a superb family upbringing . '' Eagle Scout Gerald Ford ( circled in red ) in 1929 ; Michigan Governor Fred W. Green at far left , holding hat Ford as a
center on the University of Michigan football team , 1933 Ford was involved in the Boy Scouts of America , and earned that program 's highest rank , Eagle Scout
. He is the only Eagle Scout to have ascended to the U.S. Presidency . Ford attended Grand Rapids South High School , where he was a star athlete and captain of
the football team . In 1930 , he was selected to the All - City team of the Grand Rapids City League .
</item 8>

<item 9>
Gerald Ford
Bush , Peter Dominick , Robert Griffin , Thomas Kuchel , Mel Laird , Bob Mathias , George Murphy , Dick Poff , Chuck Percy , Al Quie , Charlotte Reid , Hugh
Scott , Bill Steiger , John Tower ( 1968 ) Vacant Title next held by Donald Fraser , Scoop Jackson , Mike Mansfield , John McCormack , Patsy Mink , Ed Muskie ,
Bill Proxmire Preceded by Richard Nixon Republican nominee for President of the United States 1976 Succeeded by Ronald Reagan Political offices Preceded by
Spiro Agnew Vice President of the United States 1973 -- 1974 Succeeded by Nelson Rockefeller Preceded by Richard Nixon President of the United States 1974 --
1977 Succeeded by Jimmy Carter Diplomatic posts Preceded by Valéry Giscard d'Estaing Chair of the Group of Seven 1976 Succeeded by James Callaghan Honorary
titles Preceded by Ronald Reagan Oldest living President of the United States 2004 -- 2006 Succeeded by George H.W.
</item 9>

<item 10>
President of the United States
Educational site on the American presidency `` Presidents ' Occupations '' . Retrieved August 20 , 2007 . Listing of every president 's occupations before and
after becoming the commander in chief `` The Masonic Presidents Tour '' . The Masonic Library and Museum of Pennsylvania . Archived from the original on October
12 , 2014 . Retrieved October 7 , 2005 . Brief histories of the Masonic careers of presidents who were members of the Freemasons `` The Presidents '' . American
Experience . Retrieved March 4 , 2007 . PBS site on the American presidency Presidents of the United States : Resource Guides from the Library of Congress
Shapell Manuscript Foundation . Images of documents written by U.S.
</item 10>
=== END ITEM id=eaf0218af501256b ===

=== ITEM id=f71391230dc7f0ea ===
Gold sentences and facts:
Sentence 0: The poem survives in a single manuscript , the Cotton Nero A.x. , which also includes three religious narrative poems : Pearl , Purity and Patience . All are thought to have been written by the same unknown author , dubbed the `` Pearl Poet '' or `` Gawain Poet '' , since all four are written in a North West Midland dialect of Middle English .
  Fact 0.0: The poem survives in a single manuscript, the Cotton Nero A.x.
  Fact 0.1: The Cotton Nero A.x. manuscript also includes three religious narrative poems: Pearl, Purity and Patience.
  Fact 0.2: Pearl, Purity and Patience are thought to have been written by the same unknown author.
  Fact 0.3: The unknown author is dubbed the 'Pearl Poet' or 'Gawain Poet'.
  Fact 0.4: The 'Pearl Poet' or 'Gawain Poet' name is given because all four poems are written in a North West Midland dialect of Middle English.
  Fact 0.5: All four poems are written in a North West Midland dialect of Middle English.

Retrieved items:
<item 1>
Sir Gawain and the Green Knight
The Independent on Sunday . Retrieved 2016 - 03 - 01 . External links ( edit ) Find more aboutSir Gawain and the Green Knightat Wikipedia 's sister projects
Definitions from Wiktionary Media from Wikimedia Commons News from Wikinews Quotations from Wikiquote Texts from Wikisource Learning resources from Wikiversity
Online texts High - resolution , full - sized scan of entire manuscript Sir Gawain and the Green Knight at Archive.org JRR Tolkien and EV Gordon 's Edition The
poem in Middle English Sir Gawain and the Green Knight public domain audiobook at LibriVox ( various translations ) Google Books results General information The
Gawain / Pearl Poet from the University of Calgary The Camelot Project Info on Sir Gawain Luminarium SGGK Website Oxford Bibliographies : Bibliography on Sir
Gawain and the Green Knight BBC documentary tour of locations thought to be found in the poem ... Holywell , the River Mersey , the ' Wild Wirral ' , the Peak
District , the Roaches , and Lud Chapel Sir Gawain Introduction - Article introducing various translations and adaptations of Sir Gawain The Pearl Poet Works
Sir Gawain and the Green Knight Pearl Patience Cleanness St.
</item 1>

<item 2>
Sir Gawain and the Green Knight
Still , the connection made by the copyist to the Order is not extraordinary . Christian interpretations ( edit ) Scholars have pointed out parallels between
the girdle Bertilak 's wife offers Gawain , and the fruit Eve offered to Adam in the Biblical Garden of Eden . ( Adam and Eve Lucas Cranach , ca . 1513 ) The
poem is in many ways deeply Christian , with frequent references to the fall of Adam and Eve and to Jesus Christ . Scholars have debated the depth of the
Christian elements within the poem by looking at it in the context of the age in which it was written , coming up with varying views as to what represents a
Christian element of the poem and what does not . For example , some critics compare Sir Gawain to the other three poems of the Gawain manuscript . Each has a
heavily Christian theme , causing scholars to interpret Gawain similarly . Comparing it to the poem Cleanness ( also known as Purity ) , for example , they see
it as a story of the apocalyptic fall of a civilisation , in Gawain 's case , Camelot .
</item 2>

<item 3>
Sir Gawain and the Green Knight
Other modern translations include those by Larry Benson , Brian Stone , James Winny , Helen Cooper , W.S. Merwin , Jacob Rosenberg , William Vantuono , Joseph
Glaser , Bernard O'Donoghue , John Gardner , and Francis Ingledew . A new and illustrated translation of the work by illustrator and historian Michael Smith is
currently being crowdsourced by publisher Unbound with a planned publication date of 2018 . Sword of the Valiant ( 1984 ) , one of two film adaptations ,
starring Sean Connery as the Green Knight Film and television ( edit ) The poem has been adapted to film twice , on both occasions by writer - director Stephen
Weeks : first as Gawain and the Green Knight in 1973 and again in 1984 as Sword of the Valiant : The Legend of Sir Gawain and the Green Knight , featuring Miles
O'Keeffe as Gawain and Sean Connery as the Green Knight . Both films have been criticised for deviating from the poem 's plot . Also , Bertilak and the Green
Knight are never connected .
</item 3>

<item 4>
Sir Gawain and the Green Knight
Sir Gawain and the Green Knight - wikipedia Sir Gawain and the Green Knight Jump to : navigation , search For the 1973 film adaptation , see Gawain and the
Green Knight ( film ) . Sir Gawain and the Green Knight First page of survived manuscript , circa 14th century . Author Anonymous Country Kingdom of England
Language Middle English Genre Poem , chivalric romance , Arthurian and alliterative verse . Publication date 14th century Sir Gawain and the Green Knight (
Middle English : Sir Gawayn and þe Grene Knyȝt ) is a late 14th - century Middle English chivalric romance . It is one of the best known Arthurian stories ,
with its plot combining two types of folklore motifs , the beheading game and the exchange of winnings . The Green Knight is interpreted by some as a
representation of the Green Man of folklore and by others as an allusion to Christ . Written in stanzas of alliterative verse , each of which ends in a rhyming
bob and wheel , it draws on Welsh , Irish , and English stories , as well as the French chivalric tradition .
</item 4>

<item 5>
Sir Gawain and the Green Knight
Tolkien said he was the `` most difficult character '' to interpret in Sir Gawain . His major role in Arthurian literature is that of a judge and tester of
knights , thus he is at once terrifying , friendly , and mysterious . He appears in only two other poems : The Greene Knight and King Arthur and King Cornwall .
Scholars have attempted to connect him to other mythical characters , such as Jack in the green of English tradition and to Al - Khidr , but no definitive
connection has yet been established . However , there is a possibility , as Alice Buchanan has argued , that the colour green is erroneously attributed to the
Green Knight due to the poet 's mistranslation or misunderstanding of the Irish word glas , which could either mean grey or green . In the Death of Curoi ( one
of the Irish stories from Bricriu 's Feast ) , Curoi stands in for Bertilak , and is often called `` the man of the grey mantle '' .
</item 5>

<item 6>
Sir Gawain and the Green Knight
It is an important example of a chivalric romance , which typically involves a hero who goes on a quest which tests his prowess , and it remains popular to this
day in modern English renderings from J.R.R. Tolkien , Simon Armitage , and others , as well as through film and stage adaptations . It describes how Sir Gawain
, a knight of King Arthur 's Round Table , accepts a challenge from a mysterious `` Green Knight '' who challenges any knight to strike him with his axe if he
will take a return blow in a year and a day . Gawain accepts and beheads him with his blow , at which the Green Knight stands up , picks up his head , and
reminds Gawain of the appointed time . In his struggles to keep his bargain , Gawain demonstrates chivalry and loyalty until his honour is called into question
by a test involving Lady Bertilak , the lady of the Green Knight 's castle . The poem survives in a single manuscript , the Cotton Nero A.x. , which also
includes three religious narrative poems : Pearl , Purity and Patience .
</item 6>

<item 7>
Sir Gawain and the Green Knight
Erkenwald Characters Green Knight Lady Bertilak Poetic style Bob and wheel Alliterative verse Wikiquote Wikisource texts ( hide ) Sir Gawain and the Green
Knight by Pearl Poet Characters Green Knight Gawain Lady Bertilak Guinevere Morgan le Fay Caradoc Film Gawain and the Green Knight ( 1973 ) Sword of the Valiant
( 1984 ) Literature The Squire , His Knight , and His Lady The Hollowing Opera Gawain Related The Greene Knight Honi soit qui mal y pense VIAF : 174581151 LCCN
: n79068347 GND : 4125015 - 1 SUDOC : 028245075 BNF : cb12012068r ( data ) Retrieved from `` https://en.wikipedia.org/w/index.
</item 7>

<item 8>
Sir Gawain and the Green Knight
`` King Arthur and King Cornwall '' . In Sir Gawain : Eleven Romances and Tales . Western Michigan University Medieval Institute Publications , 2000 . p. 427 .
ISBN 1 - 879288 - 59 - 1 ^ Jump up to : Lasater , Alice E. Spain to England : A Comparative Study of Arabic , European , and English Literature of the Middle
Ages . University Press of Mississippi , 1974 . ISBN 0 - 87805 - 056 - 6 Jump up ^ Rix , Michael M. `` A Re-Examination of the Castleton Garlanding '' .
Folklore . ( June 1953 ) 64.2 pp. 342 -- 344 Jump up ^ Buchanan , Alice ( June 1932 ) . `` The Irish Framework of Sir Gawain and the Green Knight '' . PLMA. 47
( 2 ) : 315 -- 338 . doi : 10.2307 / 457878 . Jump up ^ . Friedman , Albert B. Morgan le Fay in Sir Gawain and the Green Knight . Medieval Academy of America .
</item 8>

<item 9>
Sir Gawain and the Green Knight
In this interpretation , Sir Gawain is like Noah , separated from his society and warned by the Green Knight ( who is seen as God 's representative ) of the
coming doom of Camelot . Gawain , judged worthy through his test , is spared the doom of the rest of Camelot . King Arthur and his knights , however ,
misunderstand Gawain 's experience and wear garters themselves . In Cleanness the men who are saved are similarly helpless in warning their society of impending
destruction . One of the key points stressed in this interpretation is that salvation is an individual experience difficult to communicate to outsiders . In his
depiction of Camelot , the poet reveals a concern for his society , whose inevitable fall will bring about the ultimate destruction intended by God . Gawain was
written around the time of the Black Death and Peasants ' Revolt , events which convinced many people that their world was coming to an apocalyptic end and this
belief was reflected in literature and culture . However , other critics see weaknesses in this view , since the Green Knight is ultimately under the control of
Morgan le Fay , often viewed as a figure of evil in Camelot tales .
</item 9>

<item 10>
Sir Gawain and the Green Knight
( 1980 ) . Christopher Tolkien , ed . Sir Gawain and the Green Knight , Pearl , and Sir Orfeo . Random House . ISBN 978 - 0 - 345 - 27760 - 2 . Jump up ^ White
, Michael ( 2003 ) . Tolkien : A Biography . New American Library . ISBN 0 - 451 - 21242 - 8 . Jump up ^ Farley , Frank E. ( 1930 ) . `` Rev. of Banks , Sir
Gawain , and Andrew , Sir Gawain '' ( PDF ) . Speculum. 5 ( 2 ) : 222 -- 24 . doi : 10.2307 / 2847870 . JSTOR 2847870 . Jump up ^ Baragona , Alan ( 2012 ) . ``
Rev. of Howes , Borroff , Sir Gawain and the Green Knight '' . Journal of English and Germanic Philology. 111 ( 4 ) : 535 -- 38 . doi : 10.5406 / jenglgermphil.
</item 10>
=== END ITEM id=f71391230dc7f0ea ===
