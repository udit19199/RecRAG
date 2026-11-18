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

=== ITEM id=a1911892a4648ef3 ===
Gold sentences and facts:
Sentence 0: In August 2015 , the series was reinvented as Most Wanted and received a pilot order from ABC . The cast was filled out in early 2016 , and production was completed by late March . The episode was co-produced by ABC Studios and Marvel Television . In May 2016 , ABC announced that it would not pick up the pilot to series , and would look to develop other series with Marvel instead .
  Fact 0.0: In August 2015, the series was reinvented as Most Wanted.
  Fact 0.1: The series Most Wanted received a pilot order from ABC in August 2015.
  Fact 0.2: The cast was filled out in early 2016.
  Fact 0.3: Production was completed by late March.
  Fact 0.4: The episode was co-produced by ABC Studios and Marvel Television.
  Fact 0.5: In May 2016, ABC announced that it would not pick up the pilot to series.
  Fact 0.6: In May 2016, ABC announced that it would look to develop other series with Marvel instead.

Retrieved items:
<item 1>
Most Wanted (TV pilot)
The series would have been based on storylines occurring at the end of that second season , but ABC passed on the project . In August 2015 , the series was
reinvented as Most Wanted and received a pilot order from ABC . The cast was filled out in early 2016 , and production was completed by late March . The episode
was co-produced by ABC Studios and Marvel Television . In May 2016 , ABC announced that it would not pick up the pilot to series , and would look to develop
other series with Marvel instead . Contents ( hide ) 1 Premise 2 Cast and characters 2.1 Main 2.2 Guests 3 Production 3.1 Development 3.2 Writing 3.3 Casting
3.4 Filming 3.5 Marvel Cinematic Universe tie - ins 4 No pick - up and future 5 References 6 External links Premise ( edit ) Ex-spies and ex-spouses Bobbi Morse
and Lance Hunter are on the run trying to uncover a conspiracy against them . With no help from S.H.I.E.L.D.
</item 1>

<item 2>
Most Wanted (TV pilot)
^ Jump up to : Bucksbaum , Sydney ( March 18 , 2016 ) . `` How Will ' Agents of SHIELD ' Set Up Spinoff ' Marvel 's Most Wanted ' ? '' . The Hollywood Reporter
. Archived from the original on August 20 , 2016 . Retrieved August 20 , 2016 . Jump up ^ Carinacas , Peter ( February 17 , 2016 ) . `` Artisans Booked on '
Marvel 's Most Wanted , ' ' Pearl , ' ' Dirty Dancing ' and ' Stronger ' '' . Variety . Archived from the original on July 14 , 2017 . Retrieved July 14 , 2017
. Jump up ^ Carinacas , Peter ( March 15 , 2016 ) . `` Agencies Book Artisans on ' Thor : Ragnarok , ' ' Sugar Queen , ' ' Chagrin Falls ' and ' Bunker Hill '
'' . Variety . Archived from the original on March 20 , 2016 . Retrieved March 20 , 2016 .
</item 2>

<item 3>
“Most Wanted” was an unaired ABC television pilot and Marvel Cinematic Universe spin-off. The page lists it among MCU television projects and includes links to
related Marvel films and series. It was categorized as an American action, drama, science-fiction, and espionage series, and as a pilot not picked up for a
series. The page was last edited September 11, 2017.
</item 3>

<item 4>
Most Wanted (TV pilot)
^ Jump up to : `` Marvel 's Most Wanted '' . MckennaGrace.com . May 3 , 2016 . Archived from the original on July 14 , 2017 . Retrieved July 14 , 2017 . ^ Jump
up to : Frederick , Brittany . `` Who directed Spider - Man : Homecoming ? '' . Fansided . Archived from the original on July 14 , 2017 . Retrieved July 14 ,
2017 . ^ Jump up to : Walsh , S.M. ( June 6 , 2016 ) . `` ' Stitchers ' Season 3 : Meet the Cast '' . Heavy.com . Archived from the original on July 14 , 2017 .
Retrieved July 14 , 2017 . Jump up ^ Daniels , Hunter ( July 23 , 2011 ) . `` Comic - Con : Marvel TV Announces Development Slate ; Includes Live - Action AKA
Jessica Jones ( Alias ) , The Hulk , Cloak And Dagger , Mockingbird , Animated Hulk And The Agents Of S.M.A.S.H. , More '' .
</item 4>

<item 5>
*Marvel’s Most Wanted* was an unaired ABC pilot spun off from *Agents of S.H.I.E.L.D.*, created by Jeffrey Bell and Paul Zbyszewski and directed by Billy
Gierhart. Adrianne Palicki and Nick Blood reprised Bobbi Morse/Mockingbird and Lance Hunter, ex-spies and former spouses on the run without S.H.I.E.L.D. Delroy
Lindo, Fernanda Andrade, and Oded Fehr also starred. Fil
</item 5>

<item 6>
Most Wanted (TV pilot)
Retrieved August 9 , 2016 . External links ( edit ) Marvel 's Most Wanted on IMDb Marvel Cinematic Universe Feature films Phase One Iron Man The Incredible Hulk
Iron Man 2 Thor Captain America : The First Avenger The Avengers accolades Phase Two Iron Man 3 Thor : The Dark World Captain America : The Winter Soldier
Guardians of the Galaxy Avengers : Age of Ultron Ant - Man Phase Three Captain America : Civil War Doctor Strange Guardians of the Galaxy Vol. 2 Spider - Man :
Homecoming Thor : Ragnarok Black Panther Avengers : Infinity War production Ant - Man and the Wasp Untitled Avengers film production Television series ABC
Agents of S.H.I.E.L.D. episodes season 1 5 characters Agent Carter season 1 characters Inhumans `` Behold ... The Inhumans '' `` Those Who Would Destroy Us ''
Netflix Daredevil season 1 characters Jessica Jones season 1 characters Luke Cage season 1 characters Iron Fist season 1 characters The Defenders The Punisher
Hulu Runaways Freeform Cloak & Dagger Unaired pilot Most Wanted Other Short films Item 47 Agent Carter All Hail the King Digital series WHIH Newsfront Agents of
S.H.I.E.L.D.
</item 6>

<item 7>
Most Wanted (TV pilot)
Most Wanted ( TV pilot ) - wikipedia Most Wanted ( TV pilot ) Jump to : navigation , search Most Wanted Genre Action / Adventure Science fiction Superhero Drama
Created by Jeffrey Bell Paul Zbyszewski Based on Mockingbird by Gerry Conway Barry Smith Lance Hunter by Gary Friedrich Herb Trimpe Starring Adrianne Palicki
Nick Blood Delroy Lindo Fernanda Andrade Oded Fehr Country of origin United States Original language ( s ) English Production Executive producer ( s ) Jeffrey
Bell Paul Zbyszewski Jeph Loeb Jim Chory Production company ( s ) ABC Studios Marvel Television Distributor Disney -- ABC Domestic Television Release Original
network ABC Picture format 720p ( HDTV ) Audio format 5.1 surround sound Chronology Related shows Marvel 's Agents of S.H.I.E.L.D. Marvel Cinematic Universe
television series Marvel 's Most Wanted , or simply Most Wanted , is an unaired American television pilot based on the Marvel Comics characters Bobbi Morse and
Lance Hunter .
</item 7>

<item 8>
*Most Wanted* was a proposed Marvel/ABC series following ex-spies and former spouses Bobbi Morse and Lance Hunter as they evade pursuit and uncover a conspiracy
without help from S.H.I.E.L.D. Initially based on storylines from the end of *Agents of S.H.I.E.L.D.*’s second season, the project was reworked and ordered as a
pilot in August 2015. Casting was completed in early 2016, and filming wrapped by late March. ABC Studios
</item 8>

<item 9>
*Marvel’s Most Wanted* (also known as *Most Wanted*) was an unaired ABC television pilot about Marvel Comics characters Bobbi Morse (Mockingbird) and Lance
Hunter, spun off from *Agents of S.H.I.E.L.D.* Created by Jeffrey Bell and Paul Zbyszewski, it was produced by ABC Studios and Marvel Television. The
action-adventure superhero drama starred Adrianne Palicki, Nick Blood, Delroy Lindo, Fernanda Andrade, and O
</item 9>

<item 10>
Most Wanted (TV pilot)
Retrieved March 25 , 2016 . ^ Jump up to : Abrams , Natalie ( July 25 , 2014 ) . `` Agents of S.H.I.E.L.D. Reveals New Characters , Carter Adds Captain
Directors '' . TV Guide . Archived from the original on July 26 , 2014 . Retrieved July 25 , 2014 . ^ Jump up to : Andreeva , Nellie ( January 26 , 2016 ) . ``
' Marvel 's Most Wanted ' ABC Pilot Casts Delroy Lindo As Dominic Fortune '' . Deadline.com . Archived from the original on January 26 , 2016 . Retrieved
January 27 , 2016 . Jump up ^ Huver , Scott ( April 22 , 2016 ) . `` '' Marvel 's Most Wanted '' Co-Star Andrade Promises An `` Unpredictable '' Series `` .
Comic Book Resources . Archived from the original on April 22 , 2016 . Retrieved April 22 , 2016 . ^ Jump up to : `` Marvel 's Most Wanted '' .
</item 10>
=== END ITEM id=a1911892a4648ef3 ===

=== ITEM id=a5797e6a93a29144 ===
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
. TMZ . February 22 , 2014 . Retrieved March 21 , 2014 . Jump up ^ `` The Miz and Maryse are expecting their first child '' . Retrieved March 19 , 2018 . Jump
up ^ Waldman , Jon ( April 25 , 2009 ) . `` Smackdown : UK gets a strong episode of SD ! '' . Slam ! Sports . Canadian Online Explorer . Retrieved February 14 ,
2010 . Jump up ^ David , Stephens ( January 4 , 2010 ) . `` WWE Raw Results -- 1 / 4 / 10 '' . WrestleView . Retrieved April 9 , 2016 . Jump up ^ David ,
Stephens ( May 17 , 2010 ) . `` WWE Raw Results -- 5 / 17 / 10 '' . WrestleView . Retrieved April 9 , 2016 . Jump up ^ Tedesco , Mike ( December 26 , 2008 ) .
</item 2>

<item 3>
Maryse Ouellet
In 2018 , Maryse along with The Miz will star in a reality television series titled Miz & Mrs . Maryse has appeared in four WWE video games . She made her in -
game debut at WWE Smackdown vs. Raw 2010 , later appearing in WWE Smackdown vs. Raw 2011 , WWE ' 12 , and returned in WWE 2K18 . Filmography ( edit ) Film (
edit ) Year Title Role Notes Octobre Pouilleuse Film debut 2015 Sharknado 3 : Oh Hell No !
</item 3>

<item 4>
Maryse Ouellet
On June 4 , at Extreme Rules , Maryse was in Miz 's corner when he won the WWE Intercontinental Championship for the seventh time by defeating Ambrose . In
September , following her pregnancy announcement , Maryse stopped appearing on television . On January 22 , 2018 , in the Raw 25 Years special episode , Maryse
was honored as part of a segment involving women considered legends that contributed to the company 's success , including the Bella Twins , Kelly Kelly ,
Lilian Garcia , Torrie Wilson , Michelle McCool , Terri Runnels , Maria Kanellis and the Hall of Famers Jacqueline and Trish Stratus . Other media ( edit )
Maryse and Miz in July 2016 In April 2007 , she appeared alongside fellow WWE Divas Ashley , Torrie Wilson , Brooke Adams , Layla , and Kelly Kelly in Timbaland
's music video `` Throw It On Me '' featuring The Hives . Ouellet also appeared in the January 2009 issue of Muscle & Fitness , along with Eve Torres and
Michelle McCool . She and John Morrison were interviewed on Eurosport in the same year .
</item 4>

<item 5>
Maryse Ouellet
She began appearing regularly for FWE , where she commentated during women 's matches . Return to WWE ( edit ) Managing the Miz ( 2016 -- present ) ( edit )
Maryse celebrating along The Miz after winning the Intercontinental Championship at a Raw event in April 2016 On April 4 , 2016 episode of Raw , the episode
after WrestleMania 32 , Maryse returned during a WWE Intercontinental Championship match between the champion Zack Ryder and her husband , The Miz . During the
match , Maryse slapped Ryder 's father , who was in the front row , in order to distract Ryder and allowed Miz to capture the championship for the fifth time ,
thus becoming his manager in the process , where the two then became an on - screen power couple . She formally re-introduced herself three days later on the
April 7 episode of SmackDown before once more helping Miz to retain his championship over Ryder . She then began to cut various promos with Miz as his host
during the `` Miz TV '' segments , while calling themselves the `` It '' Couple .
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
Following a hiatus due to the shoot of a WWE film , Maryse returned to WWE television on the June 27 episode of Raw , where she helped Miz to retain his
championship against Kane after she faked an ankle injury . After being drafted to Smackdown as part of the 2016 WWE draft which took place on July 19 , Ouellet
made her first appearance for the brand on July 26 , where she co-hosted a Miz TV segment along her husband before being interrupted by Randy Orton , who
challenged Miz to a non-title match , in which he was subsequently defeated . At Backlash , Maryse successfully helped Miz to retain the Intercontinental
Championship by spraying something at Dolph Ziggler whilst Miz distracted the referee . The following week , SmackDown General Manager Daniel Bryan announced a
rematch for the championship due Maryse 's interference at Backlash . On September 20 , Miz retained the championship over Ziggler .
</item 7>

<item 8>
Graph query row:
work: Miz & Mrs.
start_or_premiere_date: 2018
person: The Miz
relationship: RELATED_TO
airing_event: None
airing_date_or_time: None
</item 8>

<item 9>
Maryse Ouellet
Maryse Ouellet - wikipedia Maryse Ouellet Jump to : navigation , search Maryse Ouellet Ouellet in April 2016 Birth name Maryse Ouellet ( 1983 - 01 - 21 )
January 21 , 1983 ( age 35 ) Montréal , Quebec , Canada Residence Austin , Texas , United States Spouse ( s ) The Miz ( m . 2014 ) Children Professional
wrestling career Ring name ( s ) Maryse Billed height 5 ft 8 in ( 1.73 m ) Billed weight 116 lb ( 53 kg ) Billed from Montreal , Quebec , Canada Trained by Al
Snow Dave Finlay Steve Keirn Tom Prichard Ricky Steamboat Debut August 24 , 2006 Maryse Mizanin ( / məˈriːs / ; née Ouellet ; born January 21 , 1983 ) is a
French Canadian professional wrestler , professional wrestling manager , actress , businesswoman and glamour model currently signed to WWE under the ring name
Maryse , performing on the SmackDown brand as the manager of her real - life husband The Miz .
</item 9>

<item 10>
Maryse Ouellet
E ! Online . Retrieved December 23 , 2017 . Jump up ^ `` Maryse Ouellet profile on famousVcard '' . Jump up ^ `` Maryse talks WWE release , restraining order ,
Divas Legend she wants to work with '' . Pro Wrestling Torch . January 5 , 2012 . Retrieved April 12 , 2012 . Jump up ^ `` More Diva dreams come true '' . WWE .
August 24 , 2006 . Retrieved June 27 , 2016 . Jump up ^ Nesbitt , Cherylann . `` Mike `` The Miz '' Mizanin Knows His Fiance is hotter `` . MTV . Retrieved
August 31 , 2013 . Jump up ^ Donnelly , Matthew Scott . `` Mike ' The Miz ' Mizanin ties the knot with a fellow WWE Champ '' . MTV . Retrieved March 21 , 2014 .
Jump up ^ `` WWE Superstar The Miz -- My Wedding Was Awesome ! '' . TMZ . February 22 , 2014 .
</item 10>
=== END ITEM id=a5797e6a93a29144 ===

=== ITEM id=7ab6216cb3df4b8e ===
Gold sentences and facts:
Sentence 0: Country Adult prevalence of HIV / AIDS Number of people with HIV / AIDS Annual deaths from HIV / AIDS Year of estimate Swaziland 27.20 % 220,000 3,900 2016 Lesotho 25.00 % 330,000 9,900 2016 Botswana 21.90 % 360,000 3,900 2016 South Africa 18.90 % 7,100,000 110,000 2016 Namibia 13.80 % 230,000 4,300 2016 Zimbabwe 13.50 % 1,300,000 30,000 2016 Zambia 12.40 % 1,200,000 21,000 2016 Mozambique 12.30 % 1,800,000 62,000 2016 Malawi 9.20 % 1,000,000 24,000 2016 Uganda 6.50 % 1,400,000 28,000 2016 Equatorial Guinea 6.20 % 35,000 - 2016 Kenya 5.40 % 1,600,000 36,000 2016 Tanzania 4.70 % 1,400,000 33,000 2016 Central African Republic 4.00 % 130,000 7,300 2016 Cameroon 3.80 % 560,000 29,000 2016 Gabon 3.60 % 48,000 1,500 2016 Bahamas , The 3.30 % 8,200 - 2016 Rwanda 3.10 % 220,000 3,300 2016 Congo , Republic of the 3.10 % 91,000 3,800 2016 Guinea - Bissau 3.10 % 36,000 2,000 2016 Nigeria 2.90 % 3,200,000 160,000 2016 Cote d'Ivoire 2.70 % 460,000 25,000 2016 South Sudan 2.70 % 200,000 13,000 2016 Haiti 2.10 % 150,000 4,600 2016 Togo 2.10 % 100,000 5,100 2016 Angola 1.90 % 280,000 11,000 2016 Belize 1.80 % 4,300 - 2016 Sierra Leone 1.70 % 67,000 2,800 2016 Gambia , The 1.70 % 20,000 1,100 2016 Jamaica 1.70 % 30,000 1,300 2016 Liberia 1.60 % 43,000 2,800 2016 Guyana 1.60 % 8,500 - 2016 Ghana 1.60 % 290,000 15,000 2016 Guinea 1.50 % 120,000 5,800 2016 Suriname 1.40 % 4,900 - 2016 Barbados 1.30 % 2,600 - 2016 Djibouti 1.30 % 8,600 - 2016 Chad 1.30 % 110,000 2,800 2016 Trinidad and Tobago 1.20 % 11,000 - 2016 Thailand 1.10 % 450,000 16,000 2016 Ethiopia 1.10 % 710,000 20,000 2016 Burundi 1.10 % 84,000 2,900 2016 Benin 1.00 % 67,000 2,400 2016 Mali 1.00 % 110,000 6,100 2016 Dominican Republic 1.00 % 67,000 2,200 2016 Ukraine 0.90 % 240,000 8,500 2016 Papua New Guinea 0.90 % 46,000 1,100 2016 Cabo Verde 0.80 % 2,800 - 2016 Panama 0.80 % 21,000 - 2016 Burma 0.80 % 230,000 7,800 2016 Burkina Faso 0.80 % 95,000 3,100 2016 Latvia 0.70 % 6,600 - 2016 Congo , Democratic Republic of 0.70 % 370,000 19,000 2016 Cambodia 0.60 % 71,000 1,800 2016 Venezuela 0.60 % 120,000 2,500 2016 Moldova 0.60 % 15,000 - 2016 Eritrea 0.60 % 15,000 - 2016 El Salvador 0.60 % 24,000 - 2016 Uruguay 0.60 % 12,000 - 2016 Brazil 0.60 % 830,000 14,000 2016 Georgia 0.50 % 12,000 - 2016 Mauritania 0.50 % 11,000 - 2016 Chile 0.50 % 61,000 - 2016 Guatemala 0.50 % 46,000 1,600 2016 Paraguay 0.50 % 19,000 - 2016 Niger 0.40 % 48,000 3,400 2016 Malaysia 0.40 % 97,000 7,000 2016 Indonesia 0.40 % 620,000 38,000 2016 Honduras 0.40 % 21,000 - 2016 Canada 0.40 % 180,000 - 2016 Cuba 0.40 % 25,000 - 2016 Costa Rica 0.40 % 13,000 - 2016 Colombia 0.40 % 120,000 2,800 2016 Vietnam 0.40 % 250,000 8,000 2016 Argentina 0.40 % 120,000 2,400 2016 Belarus 0.40 % 19,000 - 2016 USA 0.30 % 1,100,000 - 2015 Somalia 2016 Senegal 0.40 % 41,000 1,900 2016 India 0.30 % 2,100,000 62,000 2016 Ecuador 0.30 % 33,000 - 2016 Bolivia 0.30 % 19,000 - 2016 Tajikistan 0.30 % 14,000 - 2016 Peru 0.30 % 70,000 2,200 2016 Italy 0.30 % 130,000 - 2016 Laos 0.30 % 11,000 - 2016 Mexico 0.30 % 220,000 4,200 2016 Armenia 0.20 % 3,300 - 2016 Ireland 0.20 % 6,200 - 2016 Kazakhstan 0.20 % 26,000 - 2016 Kyrgyzstan 0.20 % 8,500 - 2016 Lithuania 0.20 % 2,900 - 2016 Madagascar 0.20 % 31,000 1,600 2016 Nepal 0.20 % 32,000 1,700 2016 Netherlands 0.20 % 23,000 - 2016 Nicaragua 0.20 % 8,900 - 2016 Sudan 0.20 % 56,000 3,000 2016 Sweden 0.20 % 11,000 - 2016 Albania 0.10 % 1,700 - 2016 Australia 0.10 % 25,000 - 2016 Fiji 0.10 % - - 2016 Taiwan 0.10 % 31,620 - 2016 Azerbaijan 0.10 % 9,200 - 2016 Iran 0.10 % 66,000 4,000 2016 Romania 0.10 % 16,000 - 2016 Montenegro 0.10 % - - 2016 Morocco 0.10 % 22,000 - 2016 Pakistan 0.10 % 130,000 5,500 2016 Philippines 0.10 % 56,000 - 2016 Malta 0.10 % - - 2016 Algeria - 13,000 - 2016 Bangladesh - 12,000 1,000 2016 Egypt - 11,000 - 2016 Yemen - 9,900 - 2016 Saudi Arabia - 8,200 - 2016 Afghanistan - 7,500 - 2016 Sri Lanka - 4,000 - 2016 Bulgaria - 3,500 - 2016 Czech Republic - 3,400 - 2016 Tunisia - 2,900 - 2016 Serbia - 2,700 - 2016 Lebanon - 2,200 - 2016 Croatia - 1,500 - 2016 Bosnia 0.009 % 350 - 2017
  Fact 0.0: The table gives, for each country, the adult prevalence of HIV/AIDS, the number of people with HIV/AIDS, the annual deaths from HIV/AIDS, and the year of estimate.
  Fact 0.1: The year of estimate for Somalia is 2016.
  Fact 0.2: In 2016, the adult HIV/AIDS prevalence in Swaziland was 27.20%.
  Fact 0.3: In 2016, Swaziland had 220,000 people with HIV/AIDS.
  Fact 0.4: In 2016, there were 3,900 annual deaths from HIV/AIDS in Swaziland.
  Fact 0.5: In 2016, the adult HIV/AIDS prevalence in Lesotho was 25.00%.
  Fact 0.6: In 2016, Lesotho had 330,000 people with HIV/AIDS.
  Fact 0.7: In 2016, there were 9,900 annual deaths from HIV/AIDS in Lesotho.
  Fact 0.8: In 2016, the adult HIV/AIDS prevalence in Botswana was 21.90%.
  Fact 0.9: In 2016, Botswana had 360,000 people with HIV/AIDS.
  Fact 0.10: In 2016, there were 3,900 annual deaths from HIV/AIDS in Botswana.
  Fact 0.11: In 2016, the adult HIV/AIDS prevalence in South Africa was 18.90%.
  Fact 0.12: In 2016, South Africa had 7,100,000 people with HIV/AIDS.
  Fact 0.13: In 2016, there were 110,000 annual deaths from HIV/AIDS in South Africa.
  Fact 0.14: In 2016, the adult HIV/AIDS prevalence in Namibia was 13.80%.
  Fact 0.15: In 2016, Namibia had 230,000 people with HIV/AIDS.
  Fact 0.16: In 2016, there were 4,300 annual deaths from HIV/AIDS in Namibia.
  Fact 0.17: In 2016, the adult HIV/AIDS prevalence in Zimbabwe was 13.50%.
  Fact 0.18: In 2016, Zimbabwe had 1,300,000 people with HIV/AIDS.
  Fact 0.19: In 2016, there were 30,000 annual deaths from HIV/AIDS in Zimbabwe.
  Fact 0.20: In 2016, the adult HIV/AIDS prevalence in Zambia was 12.40%.
  Fact 0.21: In 2016, Zambia had 1,200,000 people with HIV/AIDS.
  Fact 0.22: In 2016, there were 21,000 annual deaths from HIV/AIDS in Zambia.
  Fact 0.23: In 2016, the adult HIV/AIDS prevalence in Mozambique was 12.30%.
  Fact 0.24: In 2016, Mozambique had 1,800,000 people with HIV/AIDS.
  Fact 0.25: In 2016, there were 62,000 annual deaths from HIV/AIDS in Mozambique.
  Fact 0.26: In 2016, the adult HIV/AIDS prevalence in Malawi was 9.20%.
  Fact 0.27: In 2016, Malawi had 1,000,000 people with HIV/AIDS.
  Fact 0.28: In 2016, there were 24,000 annual deaths from HIV/AIDS in Malawi.
  Fact 0.29: In 2016, the adult HIV/AIDS prevalence in Uganda was 6.50%.
  Fact 0.30: In 2016, Uganda had 1,400,000 people with HIV/AIDS.
  Fact 0.31: In 2016, there were 28,000 annual deaths from HIV/AIDS in Uganda.
  Fact 0.32: In 2016, the adult HIV/AIDS prevalence in Equatorial Guinea was 6.20%.
  Fact 0.33: In 2016, Equatorial Guinea had 35,000 people with HIV/AIDS.
  Fact 0.34: In 2016, the adult HIV/AIDS prevalence in Kenya was 5.40%.
  Fact 0.35: In 2016, Kenya had 1,600,000 people with HIV/AIDS.
  Fact 0.36: In 2016, there were 36,000 annual deaths from HIV/AIDS in Kenya.
  Fact 0.37: In 2016, the adult HIV/AIDS prevalence in Tanzania was 4.70%.
  Fact 0.38: In 2016, Tanzania had 1,400,000 people with HIV/AIDS.
  Fact 0.39: In 2016, there were 33,000 annual deaths from HIV/AIDS in Tanzania.
  Fact 0.40: In 2016, the adult HIV/AIDS prevalence in Central African Republic was 4.00%.
  Fact 0.41: In 2016, Central African Republic had 130,000 people with HIV/AIDS.
  Fact 0.42: In 2016, there were 7,300 annual deaths from HIV/AIDS in Central African Republic.
  Fact 0.43: In 2016, the adult HIV/AIDS prevalence in Cameroon was 3.80%.
  Fact 0.44: In 2016, Cameroon had 560,000 people with HIV/AIDS.
  Fact 0.45: In 2016, there were 29,000 annual deaths from HIV/AIDS in Cameroon.
  Fact 0.46: In 2016, the adult HIV/AIDS prevalence in Gabon was 3.60%.
  Fact 0.47: In 2016, Gabon had 48,000 people with HIV/AIDS.
  Fact 0.48: In 2016, there were 1,500 annual deaths from HIV/AIDS in Gabon.
  Fact 0.49: In 2016, the adult HIV/AIDS prevalence in Bahamas, The was 3.30%.
  Fact 0.50: In 2016, Bahamas, The had 8,200 people with HIV/AIDS.
  Fact 0.51: In 2016, the adult HIV/AIDS prevalence in Rwanda was 3.10%.
  Fact 0.52: In 2016, Rwanda had 220,000 people with HIV/AIDS.
  Fact 0.53: In 2016, there were 3,300 annual deaths from HIV/AIDS in Rwanda.
  Fact 0.54: In 2016, the adult HIV/AIDS prevalence in Congo, Republic of the was 3.10%.
  Fact 0.55: In 2016, Congo, Republic of the had 91,000 people with HIV/AIDS.
  Fact 0.56: In 2016, there were 3,800 annual deaths from HIV/AIDS in Congo, Republic of the.
  Fact 0.57: In 2016, the adult HIV/AIDS prevalence in Guinea-Bissau was 3.10%.
  Fact 0.58: In 2016, Guinea-Bissau had 36,000 people with HIV/AIDS.
  Fact 0.59: In 2016, there were 2,000 annual deaths from HIV/AIDS in Guinea-Bissau.
  Fact 0.60: In 2016, the adult HIV/AIDS prevalence in Nigeria was 2.90%.
  Fact 0.61: In 2016, Nigeria had 3,200,000 people with HIV/AIDS.
  Fact 0.62: In 2016, there were 160,000 annual deaths from HIV/AIDS in Nigeria.
  Fact 0.63: In 2016, the adult HIV/AIDS prevalence in Cote d'Ivoire was 2.70%.
  Fact 0.64: In 2016, Cote d'Ivoire had 460,000 people with HIV/AIDS.
  Fact 0.65: In 2016, there were 25,000 annual deaths from HIV/AIDS in Cote d'Ivoire.
  Fact 0.66: In 2016, the adult HIV/AIDS prevalence in South Sudan was 2.70%.
  Fact 0.67: In 2016, South Sudan had 200,000 people with HIV/AIDS.
  Fact 0.68: In 2016, there were 13,000 annual deaths from HIV/AIDS in South Sudan.
  Fact 0.69: In 2016, the adult HIV/AIDS prevalence in Haiti was 2.10%.
  Fact 0.70: In 2016, Haiti had 150,000 people with HIV/AIDS.
  Fact 0.71: In 2016, there were 4,600 annual deaths from HIV/AIDS in Haiti.
  Fact 0.72: In 2016, the adult HIV/AIDS prevalence in Togo was 2.10%.
  Fact 0.73: In 2016, Togo had 100,000 people with HIV/AIDS.
  Fact 0.74: In 2016, there were 5,100 annual deaths from HIV/AIDS in Togo.
  Fact 0.75: In 2016, the adult HIV/AIDS prevalence in Angola was 1.90%.
  Fact 0.76: In 2016, Angola had 280,000 people with HIV/AIDS.
  Fact 0.77: In 2016, there were 11,000 annual deaths from HIV/AIDS in Angola.
  Fact 0.78: In 2016, the adult HIV/AIDS prevalence in Belize was 1.80%.
  Fact 0.79: In 2016, Belize had 4,300 people with HIV/AIDS.
  Fact 0.80: In 2016, the adult HIV/AIDS prevalence in Sierra Leone was 1.70%.
  Fact 0.81: In 2016, Sierra Leone had 67,000 people with HIV/AIDS.
  Fact 0.82: In 2016, there were 2,800 annual deaths from HIV/AIDS in Sierra Leone.
  Fact 0.83: In 2016, the adult HIV/AIDS prevalence in Gambia, The was 1.70%.
  Fact 0.84: In 2016, Gambia, The had 20,000 people with HIV/AIDS.
  Fact 0.85: In 2016, there were 1,100 annual deaths from HIV/AIDS in Gambia, The.
  Fact 0.86: In 2016, the adult HIV/AIDS prevalence in Jamaica was 1.70%.
  Fact 0.87: In 2016, Jamaica had 30,000 people with HIV/AIDS.
  Fact 0.88: In 2016, there were 1,300 annual deaths from HIV/AIDS in Jamaica.
  Fact 0.89: In 2016, the adult HIV/AIDS prevalence in Liberia was 1.60%.
  Fact 0.90: In 2016, Liberia had 43,000 people with HIV/AIDS.
  Fact 0.91: In 2016, there were 2,800 annual deaths from HIV/AIDS in Liberia.
  Fact 0.92: In 2016, the adult HIV/AIDS prevalence in Guyana was 1.60%.
  Fact 0.93: In 2016, Guyana had 8,500 people with HIV/AIDS.
  Fact 0.94: In 2016, the adult HIV/AIDS prevalence in Ghana was 1.60%.
  Fact 0.95: In 2016, Ghana had 290,000 people with HIV/AIDS.
  Fact 0.96: In 2016, there were 15,000 annual deaths from HIV/AIDS in Ghana.
  Fact 0.97: In 2016, the adult HIV/AIDS prevalence in Guinea was 1.50%.
  Fact 0.98: In 2016, Guinea had 120,000 people with HIV/AIDS.
  Fact 0.99: In 2016, there were 5,800 annual deaths from HIV/AIDS in Guinea.
  Fact 0.100: In 2016, the adult HIV/AIDS prevalence in Suriname was 1.40%.
  Fact 0.101: In 2016, Suriname had 4,900 people with HIV/AIDS.
  Fact 0.102: In 2016, the adult HIV/AIDS prevalence in Barbados was 1.30%.
  Fact 0.103: In 2016, Barbados had 2,600 people with HIV/AIDS.
  Fact 0.104: In 2016, the adult HIV/AIDS prevalence in Djibouti was 1.30%.
  Fact 0.105: In 2016, Djibouti had 8,600 people with HIV/AIDS.
  Fact 0.106: In 2016, the adult HIV/AIDS prevalence in Chad was 1.30%.
  Fact 0.107: In 2016, Chad had 110,000 people with HIV/AIDS.
  Fact 0.108: In 2016, there were 2,800 annual deaths from HIV/AIDS in Chad.
  Fact 0.109: In 2016, the adult HIV/AIDS prevalence in Trinidad and Tobago was 1.20%.
  Fact 0.110: In 2016, Trinidad and Tobago had 11,000 people with HIV/AIDS.
  Fact 0.111: In 2016, the adult HIV/AIDS prevalence in Thailand was 1.10%.
  Fact 0.112: In 2016, Thailand had 450,000 people with HIV/AIDS.
  Fact 0.113: In 2016, there were 16,000 annual deaths from HIV/AIDS in Thailand.
  Fact 0.114: In 2016, the adult HIV/AIDS prevalence in Ethiopia was 1.10%.
  Fact 0.115: In 2016, Ethiopia had 710,000 people with HIV/AIDS.
  Fact 0.116: In 2016, there were 20,000 annual deaths from HIV/AIDS in Ethiopia.
  Fact 0.117: In 2016, the adult HIV/AIDS prevalence in Burundi was 1.10%.
  Fact 0.118: In 2016, Burundi had 84,000 people with HIV/AIDS.
  Fact 0.119: In 2016, there were 2,900 annual deaths from HIV/AIDS in Burundi.
  Fact 0.120: In 2016, the adult HIV/AIDS prevalence in Benin was 1.00%.
  Fact 0.121: In 2016, Benin had 67,000 people with HIV/AIDS.
  Fact 0.122: In 2016, there were 2,400 annual deaths from HIV/AIDS in Benin.
  Fact 0.123: In 2016, the adult HIV/AIDS prevalence in Mali was 1.00%.
  Fact 0.124: In 2016, Mali had 110,000 people with HIV/AIDS.
  Fact 0.125: In 2016, there were 6,100 annual deaths from HIV/AIDS in Mali.
  Fact 0.126: In 2016, the adult HIV/AIDS prevalence in Dominican Republic was 1.00%.
  Fact 0.127: In 2016, Dominican Republic had 67,000 people with HIV/AIDS.
  Fact 0.128: In 2016, there were 2,200 annual deaths from HIV/AIDS in Dominican Republic.
  Fact 0.129: In 2016, the adult HIV/AIDS prevalence in Ukraine was 0.90%.
  Fact 0.130: In 2016, Ukraine had 240,000 people with HIV/AIDS.
  Fact 0.131: In 2016, there were 8,500 annual deaths from HIV/AIDS in Ukraine.
  Fact 0.132: In 2016, the adult HIV/AIDS prevalence in Papua New Guinea was 0.90%.
  Fact 0.133: In 2016, Papua New Guinea had 46,000 people with HIV/AIDS.
  Fact 0.134: In 2016, there were 1,100 annual deaths from HIV/AIDS in Papua New Guinea.
  Fact 0.135: In 2016, the adult HIV/AIDS prevalence in Cabo Verde was 0.80%.
  Fact 0.136: In 2016, Cabo Verde had 2,800 people with HIV/AIDS.
  Fact 0.137: In 2016, the adult HIV/AIDS prevalence in Panama was 0.80%.
  Fact 0.138: In 2016, Panama had 21,000 people with HIV/AIDS.
  Fact 0.139: In 2016, the adult HIV/AIDS prevalence in Burma was 0.80%.
  Fact 0.140: In 2016, Burma had 230,000 people with HIV/AIDS.
  Fact 0.141: In 2016, there were 7,800 annual deaths from HIV/AIDS in Burma.
  Fact 0.142: In 2016, the adult HIV/AIDS prevalence in Burkina Faso was 0.80%.
  Fact 0.143: In 2016, Burkina Faso had 95,000 people with HIV/AIDS.
  Fact 0.144: In 2016, there were 3,100 annual deaths from HIV/AIDS in Burkina Faso.
  Fact 0.145: In 2016, the adult HIV/AIDS prevalence in Latvia was 0.70%.
  Fact 0.146: In 2016, Latvia had 6,600 people with HIV/AIDS.
  Fact 0.147: In 2016, the adult HIV/AIDS prevalence in Congo, Democratic Republic of was 0.70%.
  Fact 0.148: In 2016, Congo, Democratic Republic of had 370,000 people with HIV/AIDS.
  Fact 0.149: In 2016, there were 19,000 annual deaths from HIV/AIDS in Congo, Democratic Republic of.
  Fact 0.150: In 2016, the adult HIV/AIDS prevalence in Cambodia was 0.60%.
  Fact 0.151: In 2016, Cambodia had 71,000 people with HIV/AIDS.
  Fact 0.152: In 2016, there were 1,800 annual deaths from HIV/AIDS in Cambodia.
  Fact 0.153: In 2016, the adult HIV/AIDS prevalence in Venezuela was 0.60%.
  Fact 0.154: In 2016, Venezuela had 120,000 people with HIV/AIDS.
  Fact 0.155: In 2016, there were 2,500 annual deaths from HIV/AIDS in Venezuela.
  Fact 0.156: In 2016, the adult HIV/AIDS prevalence in Moldova was 0.60%.
  Fact 0.157: In 2016, Moldova had 15,000 people with HIV/AIDS.
  Fact 0.158: In 2016, the adult HIV/AIDS prevalence in Eritrea was 0.60%.
  Fact 0.159: In 2016, Eritrea had 15,000 people with HIV/AIDS.
  Fact 0.160: In 2016, the adult HIV/AIDS prevalence in El Salvador was 0.60%.
  Fact 0.161: In 2016, El Salvador had 24,000 people with HIV/AIDS.
  Fact 0.162: In 2016, the adult HIV/AIDS prevalence in Uruguay was 0.60%.
  Fact 0.163: In 2016, Uruguay had 12,000 people with HIV/AIDS.
  Fact 0.164: In 2016, the adult HIV/AIDS prevalence in Brazil was 0.60%.
  Fact 0.165: In 2016, Brazil had 830,000 people with HIV/AIDS.
  Fact 0.166: In 2016, there were 14,000 annual deaths from HIV/AIDS in Brazil.
  Fact 0.167: In 2016, the adult HIV/AIDS prevalence in Georgia was 0.50%.
  Fact 0.168: In 2016, Georgia had 12,000 people with HIV/AIDS.
  Fact 0.169: In 2016, the adult HIV/AIDS prevalence in Mauritania was 0.50%.
  Fact 0.170: In 2016, Mauritania had 11,000 people with HIV/AIDS.
  Fact 0.171: In 2016, the adult HIV/AIDS prevalence in Chile was 0.50%.
  Fact 0.172: In 2016, Chile had 61,000 people with HIV/AIDS.
  Fact 0.173: In 2016, the adult HIV/AIDS prevalence in Guatemala was 0.50%.
  Fact 0.174: In 2016, Guatemala had 46,000 people with HIV/AIDS.
  Fact 0.175: In 2016, there were 1,600 annual deaths from HIV/AIDS in Guatemala.
  Fact 0.176: In 2016, the adult HIV/AIDS prevalence in Paraguay was 0.50%.
  Fact 0.177: In 2016, Paraguay had 19,000 people with HIV/AIDS.
  Fact 0.178: In 2016, the adult HIV/AIDS prevalence in Niger was 0.40%.
  Fact 0.179: In 2016, Niger had 48,000 people with HIV/AIDS.
  Fact 0.180: In 2016, there were 3,400 annual deaths from HIV/AIDS in Niger.
  Fact 0.181: In 2016, the adult HIV/AIDS prevalence in Malaysia was 0.40%.
  Fact 0.182: In 2016, Malaysia had 97,000 people with HIV/AIDS.
  Fact 0.183: In 2016, there were 7,000 annual deaths from HIV/AIDS in Malaysia.
  Fact 0.184: In 2016, the adult HIV/AIDS prevalence in Indonesia was 0.40%.
  Fact 0.185: In 2016, Indonesia had 620,000 people with HIV/AIDS.
  Fact 0.186: In 2016, there were 38,000 annual deaths from HIV/AIDS in Indonesia.
  Fact 0.187: In 2016, the adult HIV/AIDS prevalence in Honduras was 0.40%.
  Fact 0.188: In 2016, Honduras had 21,000 people with HIV/AIDS.
  Fact 0.189: In 2016, the adult HIV/AIDS prevalence in Canada was 0.40%.
  Fact 0.190: In 2016, Canada had 180,000 people with HIV/AIDS.
  Fact 0.191: In 2016, the adult HIV/AIDS prevalence in Cuba was 0.40%.
  Fact 0.192: In 2016, Cuba had 25,000 people with HIV/AIDS.
  Fact 0.193: In 2016, the adult HIV/AIDS prevalence in Costa Rica was 0.40%.
  Fact 0.194: In 2016, Costa Rica had 13,000 people with HIV/AIDS.
  Fact 0.195: In 2016, the adult HIV/AIDS prevalence in Colombia was 0.40%.
  Fact 0.196: In 2016, Colombia had 120,000 people with HIV/AIDS.
  Fact 0.197: In 2016, there were 2,800 annual deaths from HIV/AIDS in Colombia.
  Fact 0.198: In 2016, the adult HIV/AIDS prevalence in Vietnam was 0.40%.
  Fact 0.199: In 2016, Vietnam had 250,000 people with HIV/AIDS.
  Fact 0.200: In 2016, there were 8,000 annual deaths from HIV/AIDS in Vietnam.
  Fact 0.201: In 2016, the adult HIV/AIDS prevalence in Argentina was 0.40%.
  Fact 0.202: In 2016, Argentina had 120,000 people with HIV/AIDS.
  Fact 0.203: In 2016, there were 2,400 annual deaths from HIV/AIDS in Argentina.
  Fact 0.204: In 2016, the adult HIV/AIDS prevalence in Belarus was 0.40%.
  Fact 0.205: In 2016, Belarus had 19,000 people with HIV/AIDS.
  Fact 0.206: In 2015, the adult HIV/AIDS prevalence in USA was 0.30%.
  Fact 0.207: In 2015, USA had 1,100,000 people with HIV/AIDS.
  Fact 0.208: In 2016, the adult HIV/AIDS prevalence in Senegal was 0.40%.
  Fact 0.209: In 2016, Senegal had 41,000 people with HIV/AIDS.
  Fact 0.210: In 2016, there were 1,900 annual deaths from HIV/AIDS in Senegal.
  Fact 0.211: In 2016, the adult HIV/AIDS prevalence in India was 0.30%.
  Fact 0.212: In 2016, India had 2,100,000 people with HIV/AIDS.
  Fact 0.213: In 2016, there were 62,000 annual deaths from HIV/AIDS in India.
  Fact 0.214: In 2016, the adult HIV/AIDS prevalence in Ecuador was 0.30%.
  Fact 0.215: In 2016, Ecuador had 33,000 people with HIV/AIDS.
  Fact 0.216: In 2016, the adult HIV/AIDS prevalence in Bolivia was 0.30%.
  Fact 0.217: In 2016, Bolivia had 19,000 people with HIV/AIDS.
  Fact 0.218: In 2016, the adult HIV/AIDS prevalence in Tajikistan was 0.30%.
  Fact 0.219: In 2016, Tajikistan had 14,000 people with HIV/AIDS.
  Fact 0.220: In 2016, the adult HIV/AIDS prevalence in Peru was 0.30%.
  Fact 0.221: In 2016, Peru had 70,000 people with HIV/AIDS.
  Fact 0.222: In 2016, there were 2,200 annual deaths from HIV/AIDS in Peru.
  Fact 0.223: In 2016, the adult HIV/AIDS prevalence in Italy was 0.30%.
  Fact 0.224: In 2016, Italy had 130,000 people with HIV/AIDS.
  Fact 0.225: In 2016, the adult HIV/AIDS prevalence in Laos was 0.30%.
  Fact 0.226: In 2016, Laos had 11,000 people with HIV/AIDS.
  Fact 0.227: In 2016, the adult HIV/AIDS prevalence in Mexico was 0.30%.
  Fact 0.228: In 2016, Mexico had 220,000 people with HIV/AIDS.
  Fact 0.229: In 2016, there were 4,200 annual deaths from HIV/AIDS in Mexico.
  Fact 0.230: In 2016, the adult HIV/AIDS prevalence in Armenia was 0.20%.
  Fact 0.231: In 2016, Armenia had 3,300 people with HIV/AIDS.
  Fact 0.232: In 2016, the adult HIV/AIDS prevalence in Ireland was 0.20%.
  Fact 0.233: In 2016, Ireland had 6,200 people with HIV/AIDS.
  Fact 0.234: In 2016, the adult HIV/AIDS prevalence in Kazakhstan was 0.20%.
  Fact 0.235: In 2016, Kazakhstan had 26,000 people with HIV/AIDS.
  Fact 0.236: In 2016, the adult HIV/AIDS prevalence in Kyrgyzstan was 0.20%.
  Fact 0.237: In 2016, Kyrgyzstan had 8,500 people with HIV/AIDS.
  Fact 0.238: In 2016, the adult HIV/AIDS prevalence in Lithuania was 0.20%.
  Fact 0.239: In 2016, Lithuania had 2,900 people with HIV/AIDS.
  Fact 0.240: In 2016, the adult HIV/AIDS prevalence in Madagascar was 0.20%.
  Fact 0.241: In 2016, Madagascar had 31,000 people with HIV/AIDS.
  Fact 0.242: In 2016, there were 1,600 annual deaths from HIV/AIDS in Madagascar.
  Fact 0.243: In 2016, the adult HIV/AIDS prevalence in Nepal was 0.20%.
  Fact 0.244: In 2016, Nepal had 32,000 people with HIV/AIDS.
  Fact 0.245: In 2016, there were 1,700 annual deaths from HIV/AIDS in Nepal.
  Fact 0.246: In 2016, the adult HIV/AIDS prevalence in Netherlands was 0.20%.
  Fact 0.247: In 2016, Netherlands had 23,000 people with HIV/AIDS.
  Fact 0.248: In 2016, the adult HIV/AIDS prevalence in Nicaragua was 0.20%.
  Fact 0.249: In 2016, Nicaragua had 8,900 people with HIV/AIDS.
  Fact 0.250: In 2016, the adult HIV/AIDS prevalence in Sudan was 0.20%.
  Fact 0.251: In 2016, Sudan had 56,000 people with HIV/AIDS.
  Fact 0.252: In 2016, there were 3,000 annual deaths from HIV/AIDS in Sudan.
  Fact 0.253: In 2016, the adult HIV/AIDS prevalence in Sweden was 0.20%.
  Fact 0.254: In 2016, Sweden had 11,000 people with HIV/AIDS.
  Fact 0.255: In 2016, the adult HIV/AIDS prevalence in Albania was 0.10%.
  Fact 0.256: In 2016, Albania had 1,700 people with HIV/AIDS.
  Fact 0.257: In 2016, the adult HIV/AIDS prevalence in Australia was 0.10%.
  Fact 0.258: In 2016, Australia had 25,000 people with HIV/AIDS.
  Fact 0.259: In 2016, the adult HIV/AIDS prevalence in Fiji was 0.10%.
  Fact 0.260: In 2016, the adult HIV/AIDS prevalence in Taiwan was 0.10%.
  Fact 0.261: In 2016, Taiwan had 31,620 people with HIV/AIDS.
  Fact 0.262: In 2016, the adult HIV/AIDS prevalence in Azerbaijan was 0.10%.
  Fact 0.263: In 2016, Azerbaijan had 9,200 people with HIV/AIDS.
  Fact 0.264: In 2016, the adult HIV/AIDS prevalence in Iran was 0.10%.
  Fact 0.265: In 2016, Iran had 66,000 people with HIV/AIDS.
  Fact 0.266: In 2016, there were 4,000 annual deaths from HIV/AIDS in Iran.
  Fact 0.267: In 2016, the adult HIV/AIDS prevalence in Romania was 0.10%.
  Fact 0.268: In 2016, Romania had 16,000 people with HIV/AIDS.
  Fact 0.269: In 2016, the adult HIV/AIDS prevalence in Montenegro was 0.10%.
  Fact 0.270: In 2016, the adult HIV/AIDS prevalence in Morocco was 0.10%.
  Fact 0.271: In 2016, Morocco had 22,000 people with HIV/AIDS.
  Fact 0.272: In 2016, the adult HIV/AIDS prevalence in Pakistan was 0.10%.
  Fact 0.273: In 2016, Pakistan had 130,000 people with HIV/AIDS.
  Fact 0.274: In 2016, there were 5,500 annual deaths from HIV/AIDS in Pakistan.
  Fact 0.275: In 2016, the adult HIV/AIDS prevalence in Philippines was 0.10%.
  Fact 0.276: In 2016, Philippines had 56,000 people with HIV/AIDS.
  Fact 0.277: In 2016, the adult HIV/AIDS prevalence in Malta was 0.10%.
  Fact 0.278: In 2016, Algeria had 13,000 people with HIV/AIDS.
  Fact 0.279: In 2016, Bangladesh had 12,000 people with HIV/AIDS.
  Fact 0.280: In 2016, there were 1,000 annual deaths from HIV/AIDS in Bangladesh.
  Fact 0.281: In 2016, Egypt had 11,000 people with HIV/AIDS.
  Fact 0.282: In 2016, Yemen had 9,900 people with HIV/AIDS.
  Fact 0.283: In 2016, Saudi Arabia had 8,200 people with HIV/AIDS.
  Fact 0.284: In 2016, Afghanistan had 7,500 people with HIV/AIDS.
  Fact 0.285: In 2016, Sri Lanka had 4,000 people with HIV/AIDS.
  Fact 0.286: In 2016, Bulgaria had 3,500 people with HIV/AIDS.
  Fact 0.287: In 2016, Czech Republic had 3,400 people with HIV/AIDS.
  Fact 0.288: In 2016, Tunisia had 2,900 people with HIV/AIDS.
  Fact 0.289: In 2016, Serbia had 2,700 people with HIV/AIDS.
  Fact 0.290: In 2016, Lebanon had 2,200 people with HIV/AIDS.
  Fact 0.291: In 2016, Croatia had 1,500 people with HIV/AIDS.
  Fact 0.292: In 2017, the adult HIV/AIDS prevalence in Bosnia was 0.009%.
  Fact 0.293: In 2017, Bosnia had 350 people with HIV/AIDS.

Retrieved items:
<item 1>
List of countries by HIV/AIDS adult prevalence rate
AVERT . Retrieved 3 August 2017 . Jump up ^ http://factbook.lincon.com/geos/print/country/countrypdf_bf.pdf Jump up ^ `` UNAIDS.org - HIV and AIDS estimates (
2013 ) '' . Jump up ^ `` The Hindu - India has 3rd - highest number of HIV - infected people : UN '' . Jump up ^ `` The Ministry of Health has Detected ... '' .
Bhutan Observer online . 2011 - 08 - 01 . Archived from the original on 2011 - 10 - 28 . Retrieved 2011 - 11 - 21 . Jump up ^ `` CIA world factbook - country
comparison '' . Jump up ^ https://www.cdc.gov/hiv/statistics/overview/index.html Jump up ^ https://www.cdc.gov.tw/downloadfile.aspx?fid=40CE068DC7A0F111
External links ( edit ) HIV / AIDS Survey Indicators Database World Health Organisation ; Prevalence of HIV among adults aged 15 to 49 ( % ) by country World
Health Organisation ;
</item 1>

<item 2>
List of countries by HIV/AIDS adult prevalence rate
Number of deaths due to AIDS by country World Health Organisation ; Number of people ( all ages ) living with HIV by country ( show ) HIV / AIDS topics ( show )
HIV / AIDS HIV HIV structure and genome subtypes CDC classification disease progression rates HIV / AIDS diagnosis management pathophysiology prevention
research vaccination PrEP WHO disease staging system for HIV infection and disease Children Teens / Adults Countries by AIDS prevalence rate Conditions Signs
and symptoms AIDS - defining clinical condition Diffuse infiltrative lymphocytosis syndrome Lipodystrophy Nephropathy Neurocognitive disorders Pruritus
Superinfection Tuberculosis co-infection HIV Drug Resistance Database Innate resistance to HIV History History Epidemiology Multiple sex partners Timeline AIDS
Museum Timothy Ray Brown Social AIDS orphan Catholic Church and HIV / AIDS Circumcision and HIV Criminal transmission Discrimination against people Economic
impact HIV - affected community HIV / AIDS denialism Safe sex Sex education List of HIV - positive people People With AIDS Self - Empowerment Movement Culture
International AIDS Conference International AIDS Society Joint United Nations Programme on HIV / AIDS ( UNAIDS ) President 's Emergency Plan for AIDS Relief (
PEPFAR ) Treatment Action Campaign World AIDS Day Media portrayal of HIV / AIDS Misconceptions about HIV /
</item 2>

<item 3>
List of countries by HIV/AIDS adult prevalence rate
000 4,300 2016 Zimbabwe 13.50 % 1,300,000 30,000 2016 Zambia 12.40 % 1,200,000 21,000 2016 Mozambique 12.30 % 1,800,000 62,000 2016 Malawi 9.20 % 1,000,000
24,000 2016 Uganda 6.50 % 1,400,000 28,000 2016 Equatorial Guinea 6.20 % 35,000 - 2016 Kenya 5.40 % 1,600,000 36,000 2016 Tanzania 4.70 % 1,400,000 33,000 2016
Central African Republic 4.00 % 130,000 7,300 2016 Cameroon 3.80 % 560,000 29,000 2016 Gabon 3.60 % 48,000 1,500 2016 Bahamas , The 3.30 % 8,200 - 2016 Rwanda
3.10 % 220,000 3,
</item 3>

<item 4>
List of countries by HIV/AIDS adult prevalence rate
Over 10 % of all people living with HIV / AIDS reside within the region . Adult HIV prevalence exceeds 20 % in Swaziland , Botswana , and Lesotho , while an
additional six countries report adult HIV prevalence of at least 10 % . Outside Africa , the highest prevalence rate is found in the Bahamas ( 3.3 % ) . In
absolute numbers , South Africa ( 7.1 million ) , followed by Nigeria ( 3.2 million ) , and India ( 2.1 million ) had the highest HIV / AIDS number of cases by
the end of 2016 . While South Africa 's large population of HIV - positive people is attributable to its high disease prevalence ( 18.9 % , one of the highest
in the world ) , Nigeria 's is lower at 2.9 % , and India 's prevalence at 0.3 % is comparable to Canada and the UK , and is lower than the United States .
</item 4>

<item 5>
List of countries by HIV/AIDS adult prevalence rate
200 - 2016 Rwanda 3.10 % 220,000 3,300 2016 Congo , Republic of the 3.10 % 91,000 3,800 2016 Guinea - Bissau 3.10 % 36,000 2,000 2016 Nigeria 2.90 % 3,200,000
160,000 2016 Cote d'Ivoire 2.70 % 460,000 25,000 2016 South Sudan 2.70 % 200,000 13,000 2016 Haiti 2.10 % 150,000 4,600 2016 Togo 2.10 % 100,000 5,100 2016
Angola 1.90 % 280,000 11,000 2016 Belize 1.80 % 4,300 - 2016 Sierra Leone 1.70 % 67,000 2,800 2016 Gambia , The 1.70 % 20,000 1,100 2016 Jamaica 1.70 % 30,000
1,
</item 5>

<item 6>
List of countries by HIV/AIDS adult prevalence rate
100 2016 Jamaica 1.70 % 30,000 1,300 2016 Liberia 1.60 % 43,000 2,800 2016 Guyana 1.60 % 8,500 - 2016 Ghana 1.60 % 290,000 15,000 2016 Guinea 1.50 % 120,000
5,800 2016 Suriname 1.40 % 4,900 - 2016 Barbados 1.30 % 2,600 - 2016 Djibouti 1.30 % 8,600 - 2016 Chad 1.30 % 110,000 2,800 2016 Trinidad and Tobago 1.20 %
11,000 - 2016 Thailand 1.10 % 450,000 16,000 2016 Ethiopia 1.10 % 710,000 20,000 2016 Burundi 1.10 % 84,000 2,900 2016 Benin 1.00 % 67,000 2,400 2016 Mali 1.
</item 6>

<item 7>
List of countries by HIV/AIDS adult prevalence rate
Megalopolises Past and future Past population ( United Nations ) Past and future population 1 1000 1500 1600 1700 1800 1900 1907 1939 1989 2000 2005 2010 Future
population Population density Current density Current real density based on food growing capacity Growth indicators Population growth rate Natural increase
Birth rate Mortality rate Fertility rate Other demographics Age at first marriage Divorce rate Ethnic and cultural diversity level Foreign - born population
Immigrant population Linguistic diversity Median age Net migration rate Number of households Sex ratio Urban population Urbanization Health Antiviral
medications for pandemic influenza HIV / AIDS adult prevalence rate Infant and under - five mortality rates Life expectancy Percentage suffering from
undernourishment Health expenditure covered by government Suicide rate Total health expenditure per capita Body Mass Index ( BMI ) Education and innovation
Bloomberg Innovation Index Education Index International Innovation Index Innovation Union Scoreboard Literacy rate Programme for the International Assessment
of Adult Competencies Progress in International Reading Literacy Study Student skills Tertiary education attainment Trends in International Mathematics and
Science Study Women 's average years in school World Intellectual Property Indicators Economic Development aid given Official Development Assistance received
Employment rate Irrigated land area Human Development
</item 7>

<item 8>
List of countries by HIV/AIDS adult prevalence rate
00 % 67,000 2,400 2016 Mali 1.00 % 110,000 6,100 2016 Dominican Republic 1.00 % 67,000 2,200 2016 Ukraine 0.90 % 240,000 8,500 2016 Papua New Guinea 0.90 %
46,000 1,100 2016 Cabo Verde 0.80 % 2,800 - 2016 Panama 0.80 % 21,000 - 2016 Burma 0.80 % 230,000 7,800 2016 Burkina Faso 0.80 % 95,000 3,100 2016 Latvia 0.70 %
6,600 - 2016 Congo , Democratic Republic of 0.70 % 370,000 19,000 2016 Cambodia 0.60 % 71,000 1,800 2016 Venezuela 0.60 % 120,000 2,500 2016 Moldova 0.60 %
15,000 - 2016 Eritrea 0.
</item 8>

<item 9>
List of countries by HIV/AIDS adult prevalence rate
40 % 25,000 - 2016 Costa Rica 0.40 % 13,000 - 2016 Colombia 0.40 % 120,000 2,800 2016 Vietnam 0.40 % 250,000 8,000 2016 Argentina 0.40 % 120,000 2,400 2016
Belarus 0.40 % 19,000 - 2016 USA 0.30 % 1,100,000 - 2015 Somalia 2016 Senegal 0.40 % 41,000 1,900 2016 India 0.30 % 2,100,000 62,000 2016 Ecuador 0.30 % 33,000
- 2016 Bolivia 0.30 % 19,000 - 2016 Tajikistan 0.30 % 14,000 - 2016 Peru 0.30 % 70,000 2,200 2016 Italy 0.30 % 130,000 - 2016 Laos 0.30 % 11,000 - 2016 Mexico
0.30 % 220,
</item 9>

<item 10>
List of countries by HIV/AIDS adult prevalence rate
List of countries by HIV / AIDS adult prevalence rate - Wikipedia List of countries by HIV / AIDS adult prevalence rate Jump to : navigation , search This
article needs to be updated . Please update this article to reflect recent events or newly available information . ( October 2017 ) AIDS and HIV prevalence 2009
No data < 0.1 % 0.1 -- 0.5 % 0.5 -- 1 % 1 -- 5 % 5 -- 15 % 15 -- 50 % The human immunodeficiency virus ( HIV ) , which causes AIDS , varies in prevalence from
nation to nation . Listed here are the prevalence rates among adults in various countries , based on data from various sources , largely the CIA World Factbook
. As of 2015 , it is estimated that there are 36.7 million people worldwide infected with HIV . The HIV pandemic is most severe in South Africa . Over 10 % of
all people living with HIV / AIDS reside within the region .
</item 10>
=== END ITEM id=7ab6216cb3df4b8e ===

=== ITEM id=af39e264ff6cf26f ===
Gold sentences and facts:
Sentence 0: Prior to 1940 the term `` jeep '' had been used as U.S. Army slang for new recruits or vehicles , but the World War II `` jeep '' that went into production in 1941 specifically tied the name to this light military 4x4 , arguably making them the oldest four - wheel drive mass - production vehicles now known as SUVs . The Jeep became the primary light 4 - wheel - drive vehicle of the United States Army and the Allies during World War II , as well as the postwar period . The term became common worldwide in the wake of the war . Doug Stewart noted : `` The spartan , cramped , and unstintingly functional jeep became the ubiquitous World War II four - wheeled personification of Yankee ingenuity and cocky , can - do determination . ''
  Fact 0.0: Before 1940, 'jeep' had been used as U.S. Army slang for new recruits or vehicles
  Fact 0.1: The World War II 'jeep' that went into production in 1941 specifically tied the name to a light military 4x4
  Fact 0.2: The World War II jeep arguably made these vehicles the oldest four-wheel drive mass-production vehicles now known as SUVs
  Fact 0.3: The Jeep became the primary light four-wheel-drive vehicle of the United States Army during World War II
  Fact 0.4: The Jeep became the primary light four-wheel-drive vehicle of the Allies during World War II
  Fact 0.5: The Jeep was the primary light four-wheel-drive vehicle of the United States Army and the Allies in the postwar period
  Fact 0.6: The term 'jeep' became common worldwide in the wake of the war
  Fact 0.7: Doug Stewart noted that the spartan, cramped, and unstintingly functional jeep became the ubiquitous World War II four-wheeled personification of Yankee ingenuity and cocky, can-do determination
Sentence 1: At least two Jeep models ( the CJ - 5 and the SJ Wagoneer ) enjoyed extraordinary three - decade production runs of a single body generation . Jeeps have since the war inspired a number of other light utility vehicles , such as the Land Rover . Many Jeep variants serving similar military and civilian roles have since been designed in other nations . In lowercase the term `` jeep '' continues to be used as a generic term for vehicles inspired by the Jeep that are suitable for use on rough terrain .
  Fact 1.0: At least two Jeep models, the CJ-5 and the SJ Wagoneer, enjoyed extraordinary three-decade production runs
  Fact 1.1: The CJ-5 and the SJ Wagoneer each had a production run of a single body generation
  Fact 1.2: Jeeps have inspired a number of other light utility vehicles since the war, such as the Land Rover
  Fact 1.3: Many Jeep variants serving similar military and civilian roles have since been designed in other nations
  Fact 1.4: In lowercase, the term 'jeep' continues to be used as a generic term
  Fact 1.5: The generic term 'jeep' refers to vehicles inspired by the Jeep that are suitable for use on rough terrain
Sentence 2: Jeep 's current product range consists solely of sport utility vehicles and off - road vehicles , but has also included pickup trucks in the past . Some of Jeep 's vehicles , such as the Grand Cherokee and the Wagoneer , reach into the luxury segment . Jeep sold 1.4 million SUVs globally in 2016 , up from 500,000 in 2008 , two - thirds of which in North America , and was Fiat - Chrysler 's best selling brand in the U.S. during the first half of 2017 . In the U.S. alone , over 2400 dealerships hold franchise rights to sell Jeep - branded vehicles , and if Jeep were spun off into a separate company , it is estimated to be worth between $22 and $33.5 billion -- slightly more than all of FCA ( US ) .
  Fact 2.0: Jeep's current product range consists solely of sport utility vehicles and off-road vehicles
  Fact 2.1: Jeep's product range has also included pickup trucks in the past
  Fact 2.2: Some of Jeep's vehicles, such as the Grand Cherokee and the Wagoneer, reach into the luxury segment
  Fact 2.3: Jeep sold 1.4 million SUVs globally in 2016
  Fact 2.4: Jeep sold 500,000 SUVs globally in 2008
  Fact 2.5: Two-thirds of the 1.4 million SUVs Jeep sold globally in 2016 were sold in North America
  Fact 2.6: Jeep was Fiat-Chrysler's best selling brand in the U.S. during the first half of 2017
  Fact 2.7: In the U.S. alone, over 2400 dealerships hold franchise rights to sell Jeep-branded vehicles
  Fact 2.8: If Jeep were spun off into a separate company, it is estimated to be worth between $22 and $33.5 billion
  Fact 2.9: That estimated value is slightly more than all of FCA (US)

Retrieved items:
<item 1>
Jeep
With its solid axles and open top , the Wrangler has been called the Jeep model that is as central to the brand 's identity as the rear - engined 911 is to
Porsche . At least two Jeep models ( the CJ - 5 and the SJ Wagoneer ) enjoyed extraordinary three - decade production runs of a single body generation . Jeeps
have since the war inspired a number of other light utility vehicles , such as the Land Rover . Many Jeep variants serving similar military and civilian roles
have since been designed in other nations . In lowercase the term `` jeep '' continues to be used as a generic term for vehicles inspired by the Jeep that are
suitable for use on rough terrain . Contents ( hide ) 1 World War II Jeeps 1.1 Development -- 1 . Bantam Reconnaissance Car 1.2 Development -- 2 . Enter Willys
and Ford 1.3 Full production -- Willys MB and Ford GPW 2 Post-war military Jeeps 2.1 CJ - V35 / U 2.2 M715 3 Jeep etymology 4 Brand , trademarks and image 4.1
Off - road abilities 5 Company history and ownership 5.
</item 1>

<item 2>
Jeep
5 Main articles : Jeep Cherokee ( XJ ) and Jeep Comanche 1984 -- 2001 XJ Cherokee 1984 -- 1990 XJ Wagoneer 1984 -- 1985 Broughwood 1984 -- 1990 Limited 1986 --
1992 Jeep Comanche ( MJ ) Jeep Wrangler 1989 Jeep Wrangler YJ Islander 1997 Jeep Wrangler TJ 2016 Jeep Wrangler JK Unlimited ( MIAS ' 16 ) Jeep Wrangler Main
article : Jeep Wrangler 1987 -- 1995 Jeep Wrangler YJ 1991 -- 1993 Renegade 1988 -- 1995 Wrangler Long -- Venezuela 1995 Wrangler Rio Grande 1997 -- 2006
Wrangler TJ 2002 TJ Se , X , Sport , Sahara models 2003 TJ Rubicon , Rubicon Tomb Raider Edition , Sahara , Sport , X , Se models , Freedom Edition 2004 -- 2006
TJ Long Wheel Base ( LJ ) Unlimited ( 15 '' Longer than a standard TJ ) Rubicon , Sport , X , Se models 2004 -- 2005 Willys Edition ( 2004 -- 1997 made ,
</item 2>

<item 3>
Jeep
Jeep 's current product range consists solely of sport utility vehicles and off - road vehicles , but has also included pickup trucks in the past . Some of Jeep
's vehicles , such as the Grand Cherokee and the Wagoneer , reach into the luxury segment . Jeep sold 1.4 million SUVs globally in 2016 , up from 500,000 in
2008 , two - thirds of which in North America , and was Fiat - Chrysler 's best selling brand in the U.S. during the first half of 2017 . In the U.S. alone ,
over 2400 dealerships hold franchise rights to sell Jeep - branded vehicles , and if Jeep were spun off into a separate company , it is estimated to be worth
between $22 and $33.5 billion -- slightly more than all of FCA ( US ) . Prior to 1940 the term `` jeep '' had been used as U.S.
</item 3>

<item 4>
Jeep
Also referred to as `` any small plane , helicopter , or gadget . '' This definition is supported by the use of the term `` jeep carrier '' to refer to the Navy
's small escort carriers . Early in 1941 , Willys - Overland demonstrated the vehicle 's off - road capability by having it drive up the steps of the United
States Capitol , driven by Willys test driver Irving `` Red '' Hausmann , who had recently heard soldiers at Fort Holabird calling it a `` jeep . '' When asked
by syndicated columnist Katharine Hillyer for the Washington Daily News ( or by a bystander , according to another account ) what it was called , Hausmann
answered , `` It 's a jeep . '' Katharine Hillyer 's article was published nationally on February 19 , 1941 , and included a picture of the vehicle with the
caption : LAWMAKERS TAKE A RIDE - With Senator Meade , of New York , at the wheel , and Representative Thomas , of New Jersey , sitting beside him , one of the
Army 's new scout cars , known as `` jeeps '' or `` quads '' , climbs up the Capitol steps in a demonstration yesterday .
</item 4>

<item 5>
Jeep
( prototype ) 1963 Jeep XM - 200 : J200 based concept for developing a low - cost vehicle for third - world countries 1965 Jeep / Renault Model H : A light 4x4
prototype based on the Renault 16 1966 FWD Concept Jeepvair : Similar to the Model H but with a Chevrolet Corvair powertrain 1970 XJ001 1970 XJ002 1971 Jeep
Cowboy : A design study using AMC 's `` compact '' automobile platform 1977 Jeep II 1979 Jeep Jeepster II 1986 Cherokee Targa : A two - door Cherokee
convertible ( later revised as Jeep Freedom show car ) 1987 Comanche Thunderchief : This vehicle was put into production later as the Comanche Eliminator 1989
Jeep Concept 1 : Evolved into the ZJ Grand Cherokee 1989 Jeep Rubicon Wrangler : This vehicle was later put in production 1990 Jeep JJ : Essentially what would
later be called the Icon 1990 Jeep Freedom : A revised Cherokee Targa 1991 Jeep Wagoneer 2000 : A large design concept 1993 Jeep Ecco 1997 Jeep Cherokee
Casablanca : A special edition of Cherokee ,
</item 5>

<item 6>
Jeep
windowless hardtop 2014 Jeep Wrangler Level Red 2014 Jeep Cherokee Dakar 2014 Jeep Wrangler MOJO 2015 Jeep Chief 2015 Jeep Wrangler Africa 2015 Jeep Wrangler
Red Rock Responder 2015 Jeep Staff Car : a tribute to Jeep 's military history starting with WWII Current models The Jeep brand currently produces five models ,
but 8 vehicles are under the brand name or use the Jeep logo : Jeep Renegade BU : Subcompact Sport Utility Vehicle Jeep Wrangler JK : Standard wheelbase Compact
Sport utility vehicle , 2 - door version JK Unlimited : Long wheelbase Mid-Size sport utility vehicle , 4 - door version J8 : Mid-Size military sport utility
vehicle ; Produced by AIL , AAV , and AEV . TJL : Compact pickup truck , 2 - door version ; Produced by AAV .
</item 6>

<item 7>
Jeep
Please consider summarizing the material while citing sources as needed . ( December 2017 ) Main article : List of Jeep vehicles Jeep CJ Main article : Jeep CJ
( Civilian Jeep ) 1982 Jeep Scrambler The CJ ( for `` Civilian Jeep '' ) series were literally the first `` Jeep '' branded vehicles produced for the civilian
public , beginning in 1945 with the CJ - 2A , followed by the CJ - 3A in 1949 and the CJ - 3B in 1953 . These early Jeeps are commonly referred to as ``
flatfenders '' because their front fenders were flat and rectangular across , like on their military forebears , the Willys MB and identical Ford GPW models .
The CJ - 4 exists only as a 1951 prototype , and is the `` missing link '' between the flat - fendered CJ - 2A and CJ - 3B and the round - fendered CJ - 5 first
introduced in 1955 .
</item 7>

<item 8>
Jeep
Please consider summarizing the material while citing sources as needed . ( April 2018 ) 1944 CJ - 1 prototype 1949 Alcoa Aluminum - bodied Jeepster Coupe (
prototype ) 1949 / 1950 X-98 prototype ; with flat fenders , but a rounded hood and grille like the CJ - 5 , it may have been the first F - head - powered Jeep
1950 CJ - 4 prototype 1950 CJ - 4 M prototype 1950 CJ - 4 MA prototype 1952 CJ Coiler : experimental design for an all independent suspension , with portal -
hub swing - axles and coil - springs 1958 DJ - 3A Pickup : Prototype pickup truck version of the DJ - 3A 1958 Jeep Creep : prototype utility vehicle ; several
versions built for tests , including a Postal rig and an aircraft tug 1959 Jeep J - 100 Malibu and Berkeley : Later developed into the Wagoneer 1960 Jeep Wide -
Trac : Concept for developing a low - cost vehicle for third - world countries 1962 The Brazilian Jeepster ( prototype ) 1963 Jeep XM - 200 : J200 based concept
for developing a
</item 8>

<item 9>
Jeep
2 - door version ; produced by AEV with the Jeep logo . 2007 -- 2018 Wrangler JK 2007 -- 2009 JK Rubicon , Sahara , X 2010 JK Rubicon , Sahara , Mountain ,
Islander , and Sport Editions 2011 70th Anniversary , Call of Duty : Black Ops , Mojave , and Oscar Mike Military ( 200 made ) Editions 2011 JK - 8 Independence
-- a MOPAR JK pick - up conversion kit , mirroring the 1980s CJ - 8 Scrambler pick - up 2013 Rubicon 10th Anniversary Edition 2013 - 2017 Brute Double Cab :
Pickup truck , 4 - door version , produced by American Expedition Vehicles 2014 Willys Wheeler Edition 2017 -- Jeep Wrangler JL Grand Cherokee Main article :
Jeep Grand Cherokee 1st generation Grand Cherokee ZJ . 2008 - 2010 WK Grand Cherokee .
</item 9>

<item 10>
Jeep
Jeep - wikipedia Jeep Jump to : navigation , search This article is about the U.S. motor manufacturing company . For the original World War II Jeep , see Willys
MB . For other uses , see Jeep ( disambiguation ) . Jeep Type Division Industry Automotive Founded 1943 trademark application 1945 first civilian Jeep launched
Headquarters Toledo , Ohio , U.S. Area served Worldwide Key people Michael Manley ( CEO of Jeep division ) Sergio Marchionne ( CEO of FCA US LLC ) Products
Sport utility vehicles Luxury vehicles Owner Fiat Chrysler Automobiles Parent Willys - Overland ( 1943 - 1953 ) Kaiser - Jeep ( 1953 - 1970 ) American Motors
Corporation ( 1970 - 1986 ) Chrysler ( 1987 - present ) Website jeep.com Jeep is a brand of American automobiles that is a division of FCA US LLC ( formerly
Chrysler Group , LLC ) , a wholly owned subsidiary of Fiat Chrysler Automobiles . Jeep has been a part of Chrysler since 1987 , when Chrysler acquired the Jeep
brand , along with remaining assets , from its previous owner : American Motors Corporation ( AMC ) .
</item 10>
=== END ITEM id=af39e264ff6cf26f ===

=== ITEM id=09d1a2acfad97ef7 ===
Gold sentences and facts:
Sentence 0: Draft lotteries were conducted again in 1970 ( for those born in 1951 ) and from 1971 to 1975 ( for 1952 to 1956 births ) . The draft numbers issued in 1972 were never used the next year to call for induction into service , because the last call was December 7 and authority to induct expired on June 30 , 1973 .
  Fact 0.0: Draft lotteries were conducted again in 1970 for men born in 1951.
  Fact 0.1: Draft lotteries were conducted from 1971 to 1975 for men born from 1952 to 1956.
  Fact 0.2: The draft numbers issued in 1972 were never used the next year to call men for induction into service.
  Fact 0.3: The last call for induction was on December 7.
  Fact 0.4: Authority to induct expired on June 30, 1973.
Sentence 1: On December 1 , 1969 , the Selective Service System of the United States conducted two lotteries to determine the order of call to military service in the Vietnam War for men born from 1944 to 1950 . These lotteries occurred during a period of conscription from just before World War II to 1973 . It was the first time a lottery system had been used to select men for military service since 1942 .
  Fact 1.0: On December 1, 1969, the Selective Service System of the United States conducted two lotteries.
  Fact 1.1: The 1969 lotteries determined the order of call to military service in the Vietnam War.
  Fact 1.2: The 1969 lotteries were for men born from 1944 to 1950.
  Fact 1.3: The 1969 lotteries occurred during a period of conscription from just before World War II to 1973.
  Fact 1.4: The 1969 lotteries were the first time a lottery system had been used to select men for military service since 1942.

Retrieved items:
<item 1>
Draft lottery (1969)
Draft lottery ( 1969 ) - wikipedia Draft lottery ( 1969 ) Jump to : navigation , search This article includes a list of references , but its sources remain
unclear because it has insufficient inline citations . Please help to improve this article by introducing more precise citations . ( October 2017 ) ( Learn how
and when to remove this template message ) Representative Alexander Pirnie ( R - NY ) drawing the first number . On December 1 , 1969 , the Selective Service
System of the United States conducted two lotteries to determine the order of call to military service in the Vietnam War for men born from 1944 to 1950 . These
lotteries occurred during a period of conscription from just before World War II to 1973 . It was the first time a lottery system had been used to select men
for military service since 1942 . The lottery numbers assigned in December 1969 were used during calendar year 1970 both to call for induction and to call for
physical examination , a preliminary call covering more men .
</item 1>

<item 2>
Draft lottery (1969)
The draft numbers issued in 1972 were never used the next year to call for induction into service , because the last call was December 7 and authority to induct
expired on June 30 , 1973 . The 1972 to 1975 lottery numbers were used to call some men born 1953 to 1956 for physical exams . The highest number called for a
physical was 215 ( for tables 1970 through 1976 ) . Origins and consequences ( edit ) The reason for the 1969 draft was to add personnel to a war to stop the
spread of communism in Vietnam . After World War II , Japan left Vietnamese Emperor Bao Dai in command . Ho Chi Minh , another Vietnamese leader , saw this as
an opportunity to strike and took over Hanoi . Ho Chi Minh then set up the Democratic Republic of Vietnam . Emperor Bo , in the South , set up the Vietnam State
with Saigon as the capital . Ho Chi Minh was basing his political structure off of the communist states such as Russia while Emperor Bao Dai in the south wanted
a Vietnam that was modeled after the west , like the United States .
</item 2>

<item 3>
Draft lottery (1969)
Jump up ^ 91st U.S. Congress . `` AN ACT To amend the Military Selective Service Act of 1967 ... '' ( pdf ) . United States Government Printing Office . ( Pub.
L. 91 -- 124 , 83 Stat. 220 , enacted November 26 , 1969 ) Jump up ^ Peters , Gerhard ; Woolley , John T. `` Richard Nixon : `` Executive Order 11497 - Amending
the Selective Service Regulations to Prescribe Random Selection , '' November 26 , 1969 `` . The American Presidency Project . University of California - Santa
Barbara . ^ Jump up to : `` Vietnam War Protests - Vietnam War - HISTORY.com '' . HISTORY.com . Retrieved 2017 - 12 - 05 . Jump up ^
https://www.nytimes.com/1984/09/02/us/college-enrollment-linked-to-vietnam-war.html , https://timesmachine.nytimes.com/timesmachine/1967/03/19/107186611.pdf
Jump up ^ Fisher , Anthony C. ( 1969 ) .
</item 3>

<item 4>
Draft lottery (1969)
( 1969 ) . `` The Cost of the Draft and the Cost of Ending the Draft '' . American Economic Review . 59 ( 3 ) . Jump up ^ Ifill , Gwen ( 13 February 1992 ) . ``
THE 1992 CAMPAIGN : New Hampshire ; Clinton Thanked Colonel in ' 69 For ' Saving Me From the Draft ' '' -- via NYTimes.com . ^ Jump up to : Rosenblatt , J.R. ;
Filliben , J.J. ( 1971 ) . `` Randomization and the Draft Lottery '' . Science . 171 : 306 -- 08 . doi : 10.1126 / science. 171.3968. 306 . Jump up ^ `` How the
U.S. Draft Works '' . HowStuffWorks. 2001 - 10 - 18 . Retrieved 2017 - 12 - 05 . External links ( edit ) Landscaper.net ( 2009 ) . The Military Draft and 1969
Draft Lottery for the Vietnam War . Last modified 2009 - 03 - 24 .
</item 4>

<item 5>
Draft lottery (1969)
Due to the heavy demand for military personnel , the United States required more than what the regular military could provide , causing the acceleration of the
draft . Between 1965 and 1972 the draft provided 2,215,000 service members to the U.S. military . Previously in the United States , during the War of 1812 ,
President Madison established a commission to recommend the best ways to raise military manpower ; to keep the draft or to institute a volunteer army . After
much debate within the Nixon administration and Congress , it was decided that an all - volunteer force was affordable , feasible , and would enhance the nation
's security . President Richard Nixon issued an executive order prescribing regulations for random selection by the United States Selective Service on November
26 , 1969 . In the 1960s anti-war movements started to occur in the U.S. , mainly among those on college campuses and in more leftist circles , especially those
who embraced the `` hippie '' lifestyle . College students were entitled to a deferment ( 2 - S status ) but were subject to the draft if they dropped out or
graduated . In 1967 , the number of U.S.
</item 5>

<item 6>
Draft lottery (1969)
The 1969 draft lottery only encouraged resentment of the Vietnam War and the draft . It strengthened the anti-war movement , and all over the United States ,
people decried discrimination by the draft system `` against low - education , low - income , underprivileged members of society '' . The draft lottery had
social and economic consequences because it generated resistance to military service and the resisters , draft evaders or `` draft dodgers '' , were generally
young , well - educated , healthy men . The fear of military service in Vietnam influenced many young men born in the late 1940s to join the National Guard .
These young men were aware that the National Guard would be unlikely to send its soldiers to Vietnam . Many men were unable to join the National Guard , even
though they had passed their physicals , because many state National Guards had long waiting lists just to enlist . Still others chose legal sanctions such as
imprisonment , either showing their disapproval by burning their draft cards or draft letters , or simply not presenting themselves for the military service
test . Others left the country , commonly moving to Canada .
</item 6>

<item 7>
Draft lottery (1969)
Others left the country , commonly moving to Canada . The number of American citizens who moved to Canada during the Vietnam war because of the draft is
estimated to be around 125,000 ; it is believed that about half returned to the United States after the Nixon era ( when the war was also over ) ( 1975 ) .
Modification ( edit ) Lottery procedure was improved the next year although public discontent continued to grow until `` authority to induct expired on June 30
, 1973 '' . In 1970 , covering 1951 birthdates for use during 1971 ( sometimes called the 1971 draft ) , scientists at the National Bureau of Standards prepared
78 random permutations of the numbers 1 to 366 using random numbers selected from published tables . From the 78 permutations , 25 were selected at random and
transcribed to calendars using 1 = January 1 , 2 = January 2 , ... 365 = December 31 . Those calendars were sealed in envelopes. 25 more permutations were
selected and sealed in 25 more envelopes without transcription to calendars . The two sets of 25 envelopes were furnished to the Selective Service System .
</item 7>

<item 8>
Draft lottery (1969)
Last modified 2009 - 03 - 24 . Confirmed 2011 - 05 - 26 . -- contemporary news stories , images of Official Orders ( call to physical exam , call to report ) ,
lottery results , Draft Board classifications , Vietnam troop levels , induction statistics 1917 -- 73 . David Lane ( 2003 ) . Introduction to Graphs : Clearing
Up the Draft with Graphs . Connexions. 18 July 2003 . Confirmed 2011 - 05 - 26 . -- list of draft ranks , additional analysis . Selective Service System ( 2009
) . The Vietnam Lotteries . SSS : History and Records . Last updated 2015 - 09 - 19 . Norton Starr ( 1997 ) . `` Nonrandom Risk : The 1970 Draft Lottery '' .
Journal of Statistics Education 5.2 ( 1997 ) . Confirmed 2011 - 05 - 26 . -- The online edition includes instructions for getting the data online and a lesson
plan for statistics class using the 1970 and 1971 draft lottery data .
</item 8>

<item 9>
Draft lottery (1969)
In 1967 , the number of U.S. military personnel in Vietnam was around 500,000 . The war was costing the U.S. $25 billion a year , and many of the young men
drafted were being sent to a war they wanted no part of . Martin Luther King Jr. also started to support the anti-war movement on the grounds of it being
immoral and the amount of African Americans that were being killed . November 15 , 1969 marked the largest anti war movement in the history of the United States
. This protest featured many anti-war political speakers and popular singers of the time . Many people at the time saw Richard Nixon as a liar ; when he took
office , he claimed that he would begin troop withdrawals from Vietnam immediately . After ten months of being in office , the president had yet to start
withdrawals , anyone and the people felt lied to . Later , president Nixon claimed to have been watching sports as the anti-war demonstration took place outside
the White House . The 1970s were a time of turmoil in the United States , beginning with the civil rights movement which set the standards for practices by the
anti-war movement .
</item 9>

<item 10>
Gerald Ford
President Thieu resigned on April 21 , 1975 , publicly blaming the lack of support from the United States for the fall of his country . Two days later , on
April 23 , Ford gave a speech at Tulane University . In that speech , he announced that the Vietnam War was over `` ... as far as America is concerned '' . The
announcement was met with thunderous applause . South Vietnamese refugees arrive on a U.S. Navy vessel during Operation Frequent Wind 1,373 U.S. citizens and
5,595 Vietnamese and third country nationals were evacuated from the South Vietnamese capital of Saigon during Operation Frequent Wind . In that operation ,
military and Air America helicopters took evacuees to U.S. Navy ships off - shore during an approximately 24 - hour period on April 29 to 30 , 1975 ,
immediately preceding the fall of Saigon . During the operation , so many South Vietnamese helicopters landed on the vessels taking the evacuees that some were
pushed overboard to make room for more people .
</item 10>
=== END ITEM id=09d1a2acfad97ef7 ===

=== ITEM id=7e19defdf050003a ===
Gold sentences and facts:
Sentence 0: Spike , occasionally referred to as Butch or Killer , is a stern but occasionally dumb American bulldog who is particularly disapproving of cats , but a softie when it comes to mice ( though in his debut appearance , Dog Trouble , Spike goes after both Tom and Jerry ) , and later , his son Tyke . In the shorts Jerry would often try to get Tom in trouble with Spike making him a shoo - in for a beating from the bulldog . Spike has a few weaknesses that Tom tries to capitalize upon : his possessiveness about his bone and his ticklishness . He made his first appearance in the 1942 Tom and Jerry cartoon Dog Trouble , and his first speaking role was in 1944 's The Bodyguard , where he was voiced by Billy Bletcher up until 1949 , from which point he was voiced by Daws Butler . Unlike his father Spike , Tyke does not speak . He only communicates by yapping , whimpering , growling , facial expressions and wagging his tail . In Tom and Jerry Kids , Tyke does have a speaking role in the program and is the first time that viewers were able to hear Tyke speak . Spike is very protective towards his son and gets very angry at Tom if Tyke is bothered or harmed . Although Tyke has spoken in Tom and Jerry Kids , he has laughed in one Tom and Jerry short . After Daws Butler , Maurice LaMarche , Frank Welker , John DiMaggio , Michael Donovan , Phil LaMarr and currently Rick Zieff would all perform Spike 's voice . Tyke 's vocal effects are provided by Frank Welker and speaking roles by Patric Zimmerman .
  Fact 0.0: Spike is occasionally referred to as Butch or Killer.
  Fact 0.1: Spike is a stern but occasionally dumb American bulldog.
  Fact 0.2: Spike is particularly disapproving of cats.
  Fact 0.3: Spike is a softie when it comes to mice.
  Fact 0.4: In his debut appearance, Dog Trouble, Spike goes after both Tom and Jerry.
  Fact 0.5: Spike later appears with his son, Tyke.
  Fact 0.6: In the shorts, Jerry would often try to get Tom in trouble with Spike.
  Fact 0.7: Jerry's trouble-making makes Tom a shoo-in for a beating from the bulldog Spike.
  Fact 0.8: Spike has a few weaknesses.
  Fact 0.9: Tom tries to capitalize on Spike's weaknesses.
  Fact 0.10: Spike's weaknesses include his possessiveness about his bone.
  Fact 0.11: Spike's weaknesses include his ticklishness.
  Fact 0.12: Spike made his first appearance in the 1942 Tom and Jerry cartoon Dog Trouble.
  Fact 0.13: Spike's first speaking role was in 1944's The Bodyguard.
  Fact 0.14: Billy Bletcher voiced Spike up until 1949.
  Fact 0.15: From 1949 onward, Spike was voiced by Daws Butler.
  Fact 0.16: Tyke does not speak, unlike his father Spike.
  Fact 0.17: Tyke only communicates by yapping.
  Fact 0.18: Tyke communicates by whimpering.
  Fact 0.19: Tyke communicates by growling.
  Fact 0.20: Tyke communicates by facial expressions.
  Fact 0.21: Tyke communicates by wagging his tail.
  Fact 0.22: Tyke has a speaking role in Tom and Jerry Kids.
  Fact 0.23: Tom and Jerry Kids was the first time viewers were able to hear Tyke speak.
  Fact 0.24: Spike is very protective towards his son Tyke.
  Fact 0.25: Spike gets very angry at Tom if Tyke is bothered or harmed.
  Fact 0.26: Tyke has spoken in Tom and Jerry Kids.
  Fact 0.27: Tyke has laughed in one Tom and Jerry short.
  Fact 0.28: After Daws Butler, Maurice LaMarche performed Spike's voice.
  Fact 0.29: After Daws Butler, Frank Welker performed Spike's voice.
  Fact 0.30: After Daws Butler, John DiMaggio performed Spike's voice.
  Fact 0.31: After Daws Butler, Michael Donovan performed Spike's voice.
  Fact 0.32: After Daws Butler, Phil LaMarr performed Spike's voice.
  Fact 0.33: Rick Zieff currently performs Spike's voice.
  Fact 0.34: Frank Welker provides Tyke's vocal effects.
  Fact 0.35: Patric Zimmerman provides Tyke's speaking roles.

Retrieved items:
<item 1>
List of Tom and Jerry characters
The most frequent love interest of Tom 's is Toodles Galore , who never has any dialogue in the cartoons . Despite five shorts ending with a depiction of Tom 's
apparent death , his demise is never permanent ; he even reads about his own death in a flashback in Jerry 's Diary . He appears to die in explosions in Mouse
Trouble ( after which he is seen in heaven ) , Yankee Doodle Mouse and in Safety Second , while in The Two Mouseketeers he is guillotined offscreen . Hanna -
Barbera era ( edit ) The following characters were introduced in the theatrical shorts that were directed by William Hanna and Joseph Barbera . Spike and Tyke (
edit ) Main article : Spike and Tyke ( characters ) Spike , occasionally referred to as Butch or Killer , is a stern but occasionally dumb American bulldog who
is particularly disapproving of cats , but a softie when it comes to mice ( though in his debut appearance , Dog Trouble , Spike goes after both Tom and Jerry )
, and later , his son Tyke .
</item 1>

<item 2>
List of Tom and Jerry characters
In Tom and Jerry : Spy Quest , they are seen working for Doctor Zin . They also make an appearance in Tom and Jerry : Willy Wonka and the Chocolate Factory .
They were named after the Warner Bros. 1943 short Tin Pan Alley Cats . Tin is voiced by Greg Ellis , Pan by Jess Harnell and Alley by Richard McGonagle . Butch
( dog ) ( edit ) Main article : Butch Dog ( Tex Avery ) Butch ( Originally known as Spike , until renamed as Butch to avoid confusion with Spike from the Tom
and Jerry cartoons ) is the name of another bulldog from Tex Avery 's shorts in the 1940s and 1950s . He was one of the main antagonists in Droopy shorts (
although he was replaced by Hanna - Barbera 's Spike in the 1980s ) and also had a series of his own . He appears in a number of movies e.g. Tom and Jerry and
the Wizard of Oz and Tom and Jerry : The Magic Ring . References ( edit ) Jump up ^ Deitch , Gene ( 2015 ) .
</item 2>

<item 3>
List of Tom and Jerry characters
( as a ticking time bomb set by Tom ) . Toots ( edit ) Toots is the name given to three different characters that have appeared infrequently . The first of
these is a fluffy beige cat who appears in Puss n ' Toots and The Mouse Comes to Dinner , the latter being her only speaking role . She also has a cameo in
Smitten Kitten . This version of Toots also appeared in some 1940s Tom and Jerry comics . Toots is also a different cat by the same name who appears in The Zoot
Cat and in the Tom and Jerry Tales episode `` Kitty Cat Blues . '' She has occasionally and erroneously been referred to as `` Sheikie , '' but this is actually
Tom 's nickname , as seen on a gift card in the cartoon . Finally Toots is the name of a mouse who serves as Jerry 's love interest in Blue Cat Blues . King of
France ( edit ) In some of the mouseketeer series of cartoons , Tom serves the king of France . This is an obvious reference to Louis XVI of France , who is
shown as an extremely short - tempered and cruel monarch .
</item 3>

<item 4>
List of Tom and Jerry characters
References ( edit ) Jump up ^ Deitch , Gene ( 2015 ) . Tom and Jerry ... and Gene in Tom and Jerry : The Gene Deitch Collection ( DVD ) . 20th Century Fox . Tom
and Jerry Characters Tom Cat Jerry Mouse Butch Nibbles Spike and Tyke Mammy Two Shoes Theatrical releases Tom and Jerry ( shorts ) Spike and Tyke ( shorts ) The
Movie Cameos Anchors Aweigh Dangerous When Wet DVD releases of shorts Spotlight Collection Golden Collection Chuck Jones Collection Classic Collection Deluxe
Anniversary Collection Gene Deitch Collection Television series Show ( 1975 ) Comedy Show Kids Show episodes Tales Show ( 2014 ) episodes Television specials
Hanna - Barbera 's 50th : A Yabba Dabba Doo Celebration The Mansion Cat Santa 's Little Helpers Direct - to - video films The Magic Ring Blast Off to Mars The
Fast and the Furry Shiver Me Whiskers A Nutcracker Tale Meet Sherlock Holmes The Wizard of Oz Robin Hood and His Merry Mouse Giant Adventure The Lost Dragon Spy
Quest Back to Oz Willy Wonka and the Chocolate Factory Other formats Video games Soundtrack Related William Hanna Joseph Barbera Scott Bradley Hanna - Barbera
Founders William Hanna
</item 4>

<item 5>
List of Tom and Jerry characters
Trojan Dog ' in Tom and Jerry Comedy Show ) and movies thereafter . In his first appearance , the mad scientist is the owner of Jerry ( who is also his
assistant ) and has numerous cats trapped in a prison . When Jerry picks an orange cat named Lightning , the other cats flee in terror as the scientist pulls
him out for his latest experiment . He switches the mind of the orange cat with that of a bulldog and presents him as his present to Jerry . Thin lady ( edit )
An unnamed thin white lady became Tom 's owner in Buddies Thicker Than Water . Gene Deitch opted not to use Mammy 's character in his shorts , as he felt a ``
stereotypical black housekeep '' character `` did n't work in a modern context . '' Chuck Jones would also use her in The Unshrinkable Jerry Mouse and in the
re-animated version of Saturday Evening Puss . She also appeared in the movie Tom and Jerry : The Fast and the Furry . Like many other early human characters ,
her face was never shown .
</item 5>

<item 6>
List of Tom and Jerry characters
`` Tom '' is a generic name for a male cat . He is usually but not always , portrayed as living a comfortable , or even pampered life , while Jerry ( named ``
Jinx '' in his debut appearance ) is a small , brown , house mouse who always lives in close proximity to Tom . Despite being very energetic , determined and
much larger , Tom is no match for Jerry 's wits . Jerry also possesses surprising strength for his size , approximately the equivalent of Tom 's , lifting items
such as anvils with relative ease and withstanding considerable impacts . Although cats typically chase mice to consume them , it is quite rare for Tom to
actually try to consume Jerry . Most of his attempts are just to torment or humiliate Jerry , sometimes in revenge , and sometimes to obtain a reward from a
human for catching Jerry . By the final `` fade - out '' of each cartoon , Jerry usually emerges triumphant , while Tom is shown as the loser . However , other
results may be reached . On rare occasions , Tom triumphs , usually when Jerry becomes the aggressor or when he pushes Tom a little too far .
</item 6>

<item 7>
List of Tom and Jerry characters
He is based on the character from Robin Hood tales . Yvonne jockalong ( edit ) Love interest to both Tom and Jerry , she appears in Sky Bunny and The Tennis
Menace . Sergeant farce ( edit ) A police officer who appears in See Dr. Jackal and Hide and in The Police Kitten where he is the superior officer to both Tom
and Jerry . Alley Cat Gang ( edit ) The Tom and Jerry Comedy Show ( edit ) The following characters were introduced in the television series The Tom and Jerry
Comedy Show . Droopy ( edit ) Main article : Droopy Dog A Basset Hound often with a miserable look on his face , originally introduced in the Tex Avery shorts
of the 1940s . He first made a poster cameo in Matinee Mouse . He also appeared in the Droopy and Dripple segments of Tom and Jerry Kids . He is sometimes an
adversary of Spike . He would also appear in Tom and Jerry Tales and would appear in the direct - to - video films .
</item 7>

<item 8>
List of Tom and Jerry characters
This backfired and left Mechano with a black face , while crying in frustration . He sneaks back and promptly shoots Tom , who attempts to shoot Mechano back
for his insubordination . However , the shot came out on the wrong end and Tom faints leaving Mechano laughing himself silly . He also appeared in the Tom and
Jerry Tales episode `` Tin Cat of Tomorrow '' where he is named Verminator 7000 . Mechano ( called `` Robot Cat '' ) also appeared as a fighter in the video
game , War of the Whiskers , and was voiced by Dee Bradley Baker . The St. Bernard ( edit ) The St. Bernard is a dog who first appeared in Puttin ' on the Dog
and then again in The A-Tom - Inable Snowman . Unlike other dogs in the series , he is not hostile to cats and even attends Tom medically . Eagle ( edit ) Eagle
is an Eagle that tries to steal Jerry for food from Tom .
</item 8>

<item 9>
List of Tom and Jerry characters
Like many other early human characters , her face was never shown . Chuck Jones era ( edit ) The following characters were introduced in the theatrical shorts
that were produced by Chuck Jones in the 1960s . Tiny bulldog ( edit ) This nameless mouse - sized bulldog , designed similarly to Spike the Bulldog , is Jerry
's pet dog companion in only two shorts : The Cat 's Me - Ouch and Purr - Chance to Dream . His role is similar to Spike in cartoons such as The Bodyguard , Fit
To Be Tied , and Much Ado About Mousing , which is to defend Jerry against Tom . He is presumably the oldest son of Spike before Tyke , and this could be
identified by his looks and personality that is similar to his father Spike . Unlike Spike and Tyke he has a big bite ( similar to those of Looney Tunes '
Tasmanian Devil ) . Porpoise ( edit ) A shark who tries to eat Tom and Jerry . He appears in Cannery Rodent , Puss ' n ' Boats , Filet Meow and Surf - Bored Cat
.
</item 9>

<item 10>
List of Tom and Jerry characters
His bigger role was in Timid Tabby where he finally overcomes his fear . He also appeared in Tom and Jerry Kids where his fear seems to have re-instated even
though he pretended it has not ( his name was changed to Tim in that short ) . Fluff , Muff and Puff ( edit ) Fluff , Muff and Puff are a trio of brown , black
and orange kittens that first appeared in Heavenly Puss as adorable kittens - who were murdered by drowning - in a cameo role ( that happened in Tom 's dream )
. They also appeared in Triplet Trouble as cute but extremely mischievous kittens . They also appeared in The Tom and Jerry Show . Chérie ( edit ) Chérie is a
brown / grey female mouse who serves as Jerry 's love interest . She first appeared at the end of Springtime for Thomas , and then later in Smitten Kitten ,
Touché , Pussy Cat ! ( only cartoon where she has an identical twin ) , Tom and Chérie ( the first cartoon where she is named ) and The Mouse from H.U.N.G.E.R.
( as a ticking time bomb set by Tom ) .
</item 10>
=== END ITEM id=7e19defdf050003a ===
