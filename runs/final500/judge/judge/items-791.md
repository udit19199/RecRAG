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

=== ITEM id=d80bdfea3dcce042 ===
Gold sentences and facts:
Sentence 0: The Walking Dead Cover of The Walking Dead No. 1 . Art by Tony Moore . Publication information Publisher Image Comics Schedule Monthly Format Ongoing series Genre Post-apocalyptic Publication date October 2003 -- present No. of issues 178 Creative team Created by Robert Kirkman Tony Moore Written by Robert Kirkman Artist ( s ) Tony Moore ( # 1 -- 6 ) Charlie Adlard ( # 7 -- present ) Penciller ( s ) Tony Moore ( # 1 -- 6 ) Charlie Adlard ( # 7 -- present ) Inker ( s ) Tony Moore ( # 1 -- 6 ) Charlie Adlard ( # 7 -- present ) Stefano Gaudiano ( # 115 -- present ) Letterer ( s ) Robert Kirkman ( # 1 -- 19 ) Rus Wooton ( # 20 -- present ) Colorist ( s ) Tony Moore ( # 1 -- 5 ) Cliff Rathburn ( # 6 -- present ) Editor ( s ) Aubrey Sitterson ( # 55 -- 70 ) Sina Grace ( # 67 -- 96 ) Sean Mackiewicz ( # 97 -- present )
  Fact 0.0: The Walking Dead No. 1 has a cover with art by Tony Moore.
  Fact 0.1: The Walking Dead is published by Image Comics.
  Fact 0.2: The Walking Dead has a monthly publication schedule.
  Fact 0.3: The Walking Dead is an ongoing series.
  Fact 0.4: The Walking Dead is a post-apocalyptic comic.
  Fact 0.5: The Walking Dead was published from October 2003 to the present.
  Fact 0.6: The Walking Dead has 178 issues.
  Fact 0.7: The Walking Dead was created by Robert Kirkman and Tony Moore.
  Fact 0.8: Robert Kirkman wrote The Walking Dead.
  Fact 0.9: Tony Moore was the artist for issues 1 to 6 of The Walking Dead.
  Fact 0.10: Charlie Adlard was the artist from issue 7 of The Walking Dead to the present.
  Fact 0.11: Tony Moore was the penciller for issues 1 to 6 of The Walking Dead.
  Fact 0.12: Charlie Adlard was the penciller from issue 7 of The Walking Dead to the present.
  Fact 0.13: Tony Moore was the inker for issues 1 to 6 of The Walking Dead.
  Fact 0.14: Charlie Adlard was the inker from issue 7 of The Walking Dead to the present.
  Fact 0.15: Stefano Gaudiano was an inker of The Walking Dead from issue 115 to the present.
  Fact 0.16: Robert Kirkman was the letterer of The Walking Dead for issues 1 to 19.
  Fact 0.17: Rus Wooton was the letterer of The Walking Dead from issue 20 to the present.
  Fact 0.18: Tony Moore was the colorist of The Walking Dead for issues 1 to 5.
  Fact 0.19: Cliff Rathburn was the colorist of The Walking Dead from issue 6 to the present.
  Fact 0.20: Aubrey Sitterson was an editor of The Walking Dead for issues 55 to 70.
  Fact 0.21: Sina Grace was an editor of The Walking Dead for issues 67 to 96.
  Fact 0.22: Sean Mackiewicz was an editor of The Walking Dead from issue 97 to the present.
Sentence 1: The Walking Dead debuted in 2003 , published by Image Comics , with art by Tony Moore for the first six issues and Cliff Rathburn shading the art after issue five . Charlie Adlard took over as artist on issue # 7 , after he was approached by Kirkman . Moore also did the cover art for the first twenty - four issues and the first four trade paperbacks . The remainder were done by Adlard .
  Fact 1.0: The Walking Dead debuted in 2003.
  Fact 1.1: The Walking Dead was published by Image Comics.
  Fact 1.2: Tony Moore did the art for the first six issues of The Walking Dead.
  Fact 1.3: Cliff Rathburn shaded the art of The Walking Dead after issue five.
  Fact 1.4: Charlie Adlard took over as artist of The Walking Dead on issue 7.
  Fact 1.5: Robert Kirkman approached Charlie Adlard before Adlard took over as artist on issue 7.
  Fact 1.6: Tony Moore did the cover art for the first twenty-four issues of The Walking Dead.
  Fact 1.7: Tony Moore did the cover art for the first four trade paperbacks of The Walking Dead.
  Fact 1.8: Charlie Adlard did the remainder of the cover art of The Walking Dead.

Retrieved items:
<item 1>
The Walking Dead (comic book)
17 : Something To Fear 1 - 60706 - 615 - 7 November 21 , 2012 The Walking Dead # 97 -- 102 May 2012 -- September 2012 9 5 The Walking Dead Vol. 18 : What Comes
After 1 - 60706 - 687 - 4 June 18 , 2013 The Walking Dead # 103 -- 108 October 2012 -- March 2013 The Walking Dead Vol. 19 : March To War 1 - 60706 - 818 - 4
November 13 , 2013 The Walking Dead # 109 -- 114 April 2013 -- September 2013 10 The Walking Dead Vol. 20 : All Out War ( Part One ) 1 - 60706 - 882 - 6 March
11 , 2014 The Walking Dead # 115 -- 120 October 2013 -- January 2014 The Walking Dead Vol.
</item 1>

<item 2>
The Walking Dead (comic book)
21 : All Out War ( Part Two ) 1 - 63215 - 030 - 1 July 29 , 2014 The Walking Dead # 121 -- 126 February 2014 -- April 2014 11 6 The Walking Dead Vol. 22 : A New
Beginning 1 - 63215 - 041 - 7 November 5 , 2014 The Walking Dead # 127 -- 132 May 2014 -- October 2014 The Walking Dead Vol. 23 : Whispers Into Screams 1 -
63215 - 258 - 4 May 12 , 2015 The Walking Dead # 133 -- 138 October 2014 -- March 2015 12 The Walking Dead Vol. 24 : Life And Death 1 - 63215 - 402 - 1 August
26 , 2015 The Walking Dead # 139 -- 144 March 2015 -- July 2015 The Walking Dead Vol.
</item 2>

<item 3>
The Walking Dead (comic book)
29 : Lines We Cross 1 - 53430 - 497 - 5 March 13 , 2018 The Walking Dead # 169 -- 174 July 2017 -- December 2017 15 8 Hardcovers ( edit ) All hardcovers contain
the contents of the comics , including the covers , and in some cases bonus material . The books ' trim size is larger than the paperbacks . Each hardcover
contains two story arcs from the series . Signed versions of the books are available , each limited to 310 pieces . Title ISBN Release Date Collected Material
Cover Character The Walking Dead : Book One 978 - 1582406190 July 19 , 2006 The Walking Dead # 1 -- 12 Rick The Walking Dead : Book Two 978 - 1582406985 March 7
, 2007 The Walking Dead # 13 -- 24 Michonne The Walking Dead : Book Three 978 - 1582408255 December 19 , 2007 The Walking Dead # 25 -- 36 The Governor The
Walking Dead : Book Four 978 - 1607060000 October 29 ,
</item 3>

<item 4>
The Walking Dead (comic book)
The Walking Dead v1 : Days Gone Bye , 1 : 1 - 26 ( October 8 , 2003 ) , 1071 N. Batavia St. , Suite A , Orange , CA 92867 : Image Comics Kirkman , Robert ( w )
, Moore , Tony ( p ) , Moore , Tony ( i ) , Kirkman , Robert ( let ) . The Walking Dead v1 : Days Gone Bye , 2 : 1 - 26 ( November 12 , 2003 ) , 1071 N. Batavia
St. , Suite A , Orange , CA 92867 : Image Comics Kirkman , Robert ( w ) , Moore , Tony ( p ) , Moore , Tony ( i ) , Kirkman , Robert ( let ) . The Walking Dead
v1 : Days Gone Bye , 3 : 1 - 24 ( December 10 , 2003 ) , 1071 N. Batavia St.
</item 4>

<item 5>
The Walking Dead (comic book)
2017 Jay Bonansinga Collected editions ( edit ) The series has so far been assembled into the following collections : Trade paperbacks ( edit ) The trade
paperbacks collect story arcs of six issues each , but contain only the story and none of the original cover art from the comics . Each paperback follows the
convention of having a three - word title . The zombies in the cover art for each paperback form part of a larger image if placed end to end . Each story arc is
re-released in hardcover books containing two , omnibus editions containing four , and compendiums containing eight story arcs . Title ISBN Release Date
Collected Material Issues Published Hardcover Book Number Omnibus Number Compendium Number The Walking Dead Vol. 1 : Days Gone Bye 1 - 58240 - 358 - 9 May 12 ,
2004 The Walking Dead # 1 -- 6 October 2003 -- March 2004 The Walking Dead Vol.
</item 5>

<item 6>
The Walking Dead (comic book)
2005 Collects # 1 -- 24 The Walking Dead : Volume 2 Deluxe HC 1 - 60706 - 029 - 9 February 17 , 2009 Collects # 25 -- 48 The Walking Dead : Volume 3 Deluxe HC 1
- 60706 - 330 - 1 February 2 , 2011 Collects # 49 -- 72 The Walking Dead : Volume 4 Deluxe HC 1 - 60706 - 616 - 5 December 12 , 2012 Collects # 73 -- 96 The
Walking Dead : Volume 5 Deluxe HC 1 - 63215 - 042 - 5 November 19 , 2014 Collects # 97 -- 120 The Walking Dead : Volume 6 Deluxe HC 1 - 63215 - 521 - 4 December
2 , 2015 Collects # 121 -- 144 The Walking Dead : Volume 7 Deluxe HC 1 - 53430 - 335 - 9 November 7 , 2017 Collects # 145 -- 168 Compendium editions ( edit )
Softcover compendium editions collect 48 comic issues each .
</item 6>

<item 7>
The Walking Dead (comic book)
2008 The Walking Dead # 37 -- 48 Lori and Judith The Walking Dead : Book Five 978 - 1607061717 May 5 , 2010 The Walking Dead # 49 -- 60 Abraham The Walking Dead
: Book Six 978 - 1607063278 October 26 , 2010 The Walking Dead # 61 -- 72 Carl The Walking Dead : Book Seven 978 - 1607064398 October 18 , 2011 The Walking Dead
# 73 -- 84 Rick The Walking Dead : Book Eight 978 - 1607065937 September 26 , 2012 The Walking Dead # 85 -- 96 Andrea The Walking Dead : Book Nine 978 -
1607067986 September 17 , 2013 The Walking Dead # 97 -- 108 Negan The Walking Dead : Book Ten 978 - 1632150349 September 3 , 2014 The Walking Dead # 109 -- 120
Ezekiel The Walking Dead : Book Eleven 978 - 1632152718 February 25 ,
</item 7>

<item 8>
The Walking Dead (comic book)
12 : Life Among Them 1 - 60706 - 254 - 2 August 3 , 2010 The Walking Dead # 67 -- 72 November 2009 -- May 2010 The Walking Dead Vol. 13 : Too Far Gone 1 - 60706
- 329 - 8 November 23 , 2010 The Walking Dead # 73 -- 78 June 2010 -- October 2010 7 The Walking Dead Vol. 14 : No Way Out 1 - 60706 - 392 - 1 June 22 , 2011
The Walking Dead # 79 -- 84 November 2010 -- April 2011 The Walking Dead Vol. 15 : We Find Ourselves 1 - 60706 - 440 - 5 December 27 , 2011 The Walking Dead #
85 -- 90 May 2011 -- October 2011 8 The Walking Dead Vol. 16 : A Larger World 1 - 60706 - 559 - 2 June 6 , 2012 The Walking Dead # 91 -- 96 November 2011 --
April 2012 The Walking Dead Vol.
</item 8>

<item 9>
The Walking Dead (comic book)
Charlie Adlard took over as artist on issue # 7 , after he was approached by Kirkman . Moore also did the cover art for the first twenty - four issues and the
first four trade paperbacks . The remainder were done by Adlard . When the television series premiered in October 2010 , Image Comics announced The Walking Dead
Weekly . The first 52 issues of the series began to be reprinted on January 5 , 2011 , with one issue per week for a year . The series is periodically
re-published in trade paperbacks which contain six issues each , hardcover books with twelve issues and occasional bonus material , omnibus editions of twenty -
four issues , and compendium editions of forty - eight issues . Story arcs ( edit ) Volume 1 : Days Gone Bye ( Issues 1 -- 6 ) Rick Grimes , a sheriff 's deputy
from Kentucky , is wounded in the line of duty and emerges from a coma to find the world overrun by the undead . He returns home to find his house ransacked and
his wife and son gone .
</item 9>

<item 10>
The Walking Dead (comic book)
It focuses on Rick Grimes , a Kentucky deputy who is shot in the line of duty and awakens from a coma in a zombie apocalypse that has resulted in a state - wide
quarantine . After joining with other survivors , including his loved ones , he gradually takes on the role of leader of a community as it struggles to survive
the zombie apocalypse . First issued in 2003 by publisher Image Comics , the comic is written by Kirkman with art by Moore ( issues No. 1 -- 6 ) and Charlie
Adlard ( issue No. 7 onward ) . Moore continued to do the covers through issue No. 24 . The Walking Dead received the 2007 and 2010 Eisner Award for Best
Continuing Series at San Diego Comic - Con International . The series was adapted into the AMC television series The Walking Dead , which premiered in 2010 .
The television program loosely follows the storyline of the comic book .
</item 10>
=== END ITEM id=d80bdfea3dcce042 ===

=== ITEM id=95e2f4d14477bb3e ===
Gold sentences and facts:
Sentence 0: No . overall No. in season Title Directed by Written by Original air date U.S. viewers ( millions ) `` Pilot '' Adam Davidson Robert Kirkman & Dave Erickson August 23 , 2015 ( 2015 - 08 - 23 ) 10.13 Nick awakens in a heroin den in an abandoned church , to find his girlfriend Gloria eating a corpse . While fleeing , he is hit by a car and hospitalized . The doctor tells Madison and Travis that Nick 's claims about the incident are heroin hallucinations , but Travis believes Nick after visiting the church himself . Alicia becomes more worried about Nick 's chemical dependency . The next day , school closes early due to the high levels of absenteeism and rumors of an epidemic . Nick escapes from the hospital and meets with Calvin , hoping to learn if the drugs Calvin sold him caused him to hallucinate in the church . Calvin tries to kill Nick to prevent him from exposing Calvin as a drug dealer or impugning the quality of his drugs . In the ensuing struggle , Calvin is mortally shot . After Travis and Madison arrive , the zombified Calvin attacks them . Nick runs over Calvin repeatedly with Travis ' truck , and the three watch in disbelief as the mutilated Calvin is still able to turn his head towards them . `` So Close , Yet So Far '' Adam Davidson Marco Ramirez August 30 , 2015 ( 2015 - 08 - 30 ) 8.18 After the run - in with zombified Calvin , Nick , Madison , and Travis choose to flee to the desert , and they want Alicia , Liza , and Chris to follow . Alicia finds Matt ill in his disheveled house . Travis arrives and sees a bite on Matt , who convinces Alicia to leave . The group returns to Madison 's home to gather supplies . Nick suffers from heroin withdrawal , so Madison drives to her school to get him Oxycodone . There , she encounters Tobias scavenging food . A zombified Artie tries to bite Tobias , so Madison kills Artie and drives Tobias home . Chris ' bus is trapped in a traffic jam caused by a zombie shooting by the police . He films the event and joins in a protest against that and other recent fatal police shootings , when Travis and Liza meet up with him . A riot erupts after police shoot down another zombie , but the three Manawas find refuge with the Salazars in their gated barbershop - home . Travis tells Madison to take the kids to the desert without him ; he will catch up . Alicia witnesses her zombified neighbor , Mr. Dawson , attacking Mrs. Cruz across the street , but Madison prevents her from intervening . The group inside the barbershop remains trapped , while the riot outside intensifies . `` The Dog '' Adam Davidson Jack LoGiudice September 13 , 2015 ( 2015 - 09 - 13 ) 7.19 While a riot rages outside , a mob sets fire to the store adjoining the barbershop , forcing the Salazars and Manawas to flee . The group reaches Travis ' truck and escapes , but not before Griselda is injured by a collapsing scaffold . Unable to reach a hospital , the group drives to Madison 's house , where Nick , Madison , and Alicia temporarily flee when the zombified Mr. Dawson attempts to enter , attracted by the barking dog Nick had let in . Nick leads Madison and Alicia to the Trans ' house next door , where they take a shotgun . Travis arrives and is attacked by Mr. Dawson , who is shot and killed by Daniel . All three families decide to stay the night and evacuate in the morning . Nurse Liza tends to Griselda 's injured foot but notes that Griselda will die if not treated by a doctor . Ofelia tells Daniel they should flee with Travis , but Daniel insists his family can survive alone and will join his cousin later . The next morning , as the Clarks and Manawas start driving away , the National Guard arrives and quarantines the block . While Travis says , `` It 's going to get better , '' Daniel laments that it 's , `` too late , '' as he watches a guardsman mark the neighboring house . `` Not Fade Away '' Kari Skogland Meaghan Oppenheimer September 20 , 2015 ( 2015 - 09 - 20 ) 6.62 Days after the National Guard quarantines the neighborhood into a Safe Zone , residents try to live normally . Tensions build under the military rule . Madison is stressed by extra work caused by her home 's being overcrowded and Travis ' new role as civilian liaison . Chris shows a video to Travis and Madison of a light signaling from the Dead Zone . Travis talks Doug into getting mental help . Liza medically helps neighbors . Nick steals morphine from Hector via IV drip . Ofelia kisses Adams , who was unable to get Griselda 's medicine . Madison exchanges signals with the flashing light in the Dead Zone . She sneaks outside the fence to investigate and finds evidence that the guardsmen killed civilians , even the uninfected . Travis learns that Doug has been hospitalized for his mental issues . Dr. Exner determines that Liza is not technically a nurse . Daniel warns Madison of his El Salvador experience , when the sick were taken under the guise of receiving hospitalization but instead killed . Soldiers take Griselda and Nick to a hospital , but Nick 's family protests his departure . Liza agrees to go to assist the medical team , despite not wanting to leave her son . Travis retreats to the roof and sees the signal from the Dead Zone . Seconds later , he sees and hears gunfire , followed by darkness . 5 5 `` Cobalt '' Kari Skogland David Wiener September 27 , 2015 ( 2015 - 09 - 27 ) 6.66 In a military cell , Strand bribes a guardsman to save feverish Nick from being moved . Liza helps Dr. Exner with patients at the hospital . Chris is devastated that Liza left voluntarily to help at the hospital , but Travis promises to bring her back . Madison discovers Daniel detaining Adams in the Trans ' basement . Alicia and Chris get drunk in and vandalize the abandoned home of a wealthy family . Strand recruits Nick for an escape plot . Travis convinces Moyers ' squad to take him to the hospital to check on his friends . While en route , Moyers encourages Travis to shoot a zombie , but Travis is emotionally unable to pull the trigger . The soldiers stop to assist another squad in a building infested by zombies , and most of those soldiers , including Moyers , are overcome . The few survivors flee and drop off Travis near the Safe Zone . Travis learns that Daniel tortured Adams into revealing what `` Cobalt '' means : in the morning , all civilians will be killed , and the guardsmen will evacuate the city . Griselda dies of septic shock at the hospital ; Liza shoots her brain to prevent reanimation . Daniel visits a nearby sports arena to verify Adams ' story that it was sealed with 2,000 now - zombified civilians inside . 6 6 `` The Good Man '' Stefan Schwartz Robert Kirkman & Dave Erickson October 4 , 2015 ( 2015 - 10 - 04 ) 6.86 The group drives to the National Guard 's headquarters to rescue Liza , Griselda , and Nick . Adams agrees to be their guide when let go by Travis . The group infiltrates the base after Daniel distracts the guards by leading a horde of walkers from the arena . Travis , Madison , Daniel , and Ofelia go inside , while Alicia and Chris stay behind . Meanwhile , the walkers breach the perimeter defenses and swarm the base . Travis ' group reach the holding cells and set the detainees free before reuniting with Nick , Liza , and Strand . They try to escape through the medical ward , where they discover Dr. Exner has euthanized all of the patients . Dr. Exner tells them of an escape route before presumably committing suicide . Before they can escape , the group encounters Adams , who shoots Ofelia in the arm . Enraged , Travis brutally beats Adams and leaves him for dead . Strand leads the group to his oceanside mansion , where he reveals to Nick that he owns a yacht which he plans to escape on , called the Abigail . On the beach , Liza reveals to Madison that she had been bitten during the escape . Liza pleads with Madison and Travis to euthanize her before she turns . Travis promises to protect Chris before shooting Liza .
  Fact 0.0: The table lists each episode's overall number, season number, title, director, writer, original air date, and U.S. viewers in millions.
  Fact 0.1: Episode 'Pilot' was directed by Adam Davidson and written by Robert Kirkman and Dave Erickson.
  Fact 0.2: Episode 'Pilot' originally aired on August 23, 2015.
  Fact 0.3: Episode 'Pilot' had 10.13 million U.S. viewers.
  Fact 0.4: Nick awakens in a heroin den in an abandoned church, to find his girlfriend Gloria eating a corpse.
  Fact 0.5: While fleeing, Nick is hit by a car and hospitalized.
  Fact 0.6: The doctor tells Madison and Travis that Nick's claims about the incident are heroin hallucinations.
  Fact 0.7: Travis believes Nick after Travis visits the church himself.
  Fact 0.8: Alicia becomes more worried about Nick's chemical dependency.
  Fact 0.9: The next day, school closes early due to high levels of absenteeism and rumors of an epidemic.
  Fact 0.10: Nick escapes from the hospital and meets with Calvin, hoping to learn if the drugs Calvin sold him caused him to hallucinate in the church.
  Fact 0.11: Calvin tries to kill Nick to prevent him from exposing Calvin as a drug dealer or impugning the quality of his drugs.
  Fact 0.12: In the ensuing struggle, Calvin is mortally shot.
  Fact 0.13: After Travis and Madison arrive, the zombified Calvin attacks them.
  Fact 0.14: Nick runs over Calvin repeatedly with Travis's truck.
  Fact 0.15: The three watch in disbelief as the mutilated Calvin is still able to turn his head towards them.
  Fact 0.16: Episode 'So Close, Yet So Far' was directed by Adam Davidson and written by Marco Ramirez.
  Fact 0.17: Episode 'So Close, Yet So Far' originally aired on August 30, 2015.
  Fact 0.18: Episode 'So Close, Yet So Far' had 8.18 million U.S. viewers.
  Fact 0.19: After the run-in with zombified Calvin, Nick, Madison, and Travis choose to flee to the desert.
  Fact 0.20: Nick, Madison, and Travis want Alicia, Liza, and Chris to follow them to the desert.
  Fact 0.21: Alicia finds Matt ill in his disheveled house.
  Fact 0.22: Travis arrives and sees a bite on Matt.
  Fact 0.23: Matt convinces Alicia to leave.
  Fact 0.24: The group returns to Madison's home to gather supplies.
  Fact 0.25: Nick suffers from heroin withdrawal, so Madison drives to her school to get him Oxycodone.
  Fact 0.26: At the school, Madison encounters Tobias scavenging food.
  Fact 0.27: A zombified Artie tries to bite Tobias, so Madison kills Artie and drives Tobias home.
  Fact 0.28: Chris's bus is trapped in a traffic jam caused by a zombie shooting by the police.
  Fact 0.29: Chris films the event and joins in a protest against that and other recent fatal police shootings, when Travis and Liza meet up with him.
  Fact 0.30: A riot erupts after police shoot down another zombie.
  Fact 0.31: The three Manawas find refuge with the Salazars in their gated barbershop-home.
  Fact 0.32: Travis tells Madison to take the kids to the desert without him, and he will catch up.
  Fact 0.33: Alicia witnesses her zombified neighbor, Mr. Dawson, attacking Mrs. Cruz across the street.
  Fact 0.34: Madison prevents Alicia from intervening.
  Fact 0.35: The group inside the barbershop remains trapped, while the riot outside intensifies.
  Fact 0.36: Episode 'The Dog' was directed by Adam Davidson and written by Jack LoGiudice.
  Fact 0.37: Episode 'The Dog' originally aired on September 13, 2015.
  Fact 0.38: Episode 'The Dog' had 7.19 million U.S. viewers.
  Fact 0.39: While a riot rages outside, a mob sets fire to the store adjoining the barbershop, forcing the Salazars and Manawas to flee.
  Fact 0.40: The group reaches Travis's truck and escapes, but not before Griselda is injured by a collapsing scaffold.
  Fact 0.41: Unable to reach a hospital, the group drives to Madison's house.
  Fact 0.42: Nick, Madison, and Alicia temporarily flee when the zombified Mr. Dawson attempts to enter.
  Fact 0.43: Mr. Dawson is attracted by the barking dog Nick had let in.
  Fact 0.44: Nick leads Madison and Alicia to the Trans' house next door, where they take a shotgun.
  Fact 0.45: Travis arrives and is attacked by Mr. Dawson.
  Fact 0.46: Mr. Dawson is shot and killed by Daniel.
  Fact 0.47: All three families decide to stay the night and evacuate in the morning.
  Fact 0.48: Nurse Liza tends to Griselda's injured foot but notes that Griselda will die if not treated by a doctor.
  Fact 0.49: Ofelia tells Daniel they should flee with Travis.
  Fact 0.50: Daniel insists his family can survive alone and will join his cousin later.
  Fact 0.51: The next morning, as the Clarks and Manawas start driving away, the National Guard arrives and quarantines the block.
  Fact 0.52: Travis says 'It's going to get better.'
  Fact 0.53: Daniel laments that it's 'too late' as he watches a guardsman mark the neighboring house.
  Fact 0.54: Episode 'Not Fade Away' was directed by Kari Skogland and written by Meaghan Oppenheimer.
  Fact 0.55: Episode 'Not Fade Away' originally aired on September 20, 2015.
  Fact 0.56: Episode 'Not Fade Away' had 6.62 million U.S. viewers.
  Fact 0.57: Days after the National Guard quarantines the neighborhood into a Safe Zone, residents try to live normally.
  Fact 0.58: Tensions build under the military rule.
  Fact 0.59: Madison is stressed by extra work caused by her home's being overcrowded.
  Fact 0.60: Madison is stressed by Travis's new role as civilian liaison.
  Fact 0.61: Chris shows a video to Travis and Madison of a light signaling from the Dead Zone.
  Fact 0.62: Travis talks Doug into getting mental help.
  Fact 0.63: Liza medically helps neighbors.
  Fact 0.64: Nick steals morphine from Hector via IV drip.
  Fact 0.65: Ofelia kisses Adams, who was unable to get Griselda's medicine.
  Fact 0.66: Madison exchanges signals with the flashing light in the Dead Zone.
  Fact 0.67: Madison sneaks outside the fence to investigate.
  Fact 0.68: Madison finds evidence that the guardsmen killed civilians, even the uninfected.
  Fact 0.69: Travis learns that Doug has been hospitalized for his mental issues.
  Fact 0.70: Dr. Exner determines that Liza is not technically a nurse.
  Fact 0.71: Daniel warns Madison of his El Salvador experience, when the sick were taken under the guise of receiving hospitalization but instead killed.
  Fact 0.72: Soldiers take Griselda and Nick to a hospital, but Nick's family protests his departure.
  Fact 0.73: Liza agrees to go to assist the medical team, despite not wanting to leave her son.
  Fact 0.74: Travis retreats to the roof and sees the signal from the Dead Zone.
  Fact 0.75: Seconds later, Travis sees and hears gunfire, followed by darkness.
  Fact 0.76: Episode 'Cobalt' is overall number 5 and season number 5.
  Fact 0.77: Episode 'Cobalt' was directed by Kari Skogland and written by David Wiener.
  Fact 0.78: Episode 'Cobalt' originally aired on September 27, 2015.
  Fact 0.79: Episode 'Cobalt' had 6.66 million U.S. viewers.
  Fact 0.80: In a military cell, Strand bribes a guardsman to save feverish Nick from being moved.
  Fact 0.81: Liza helps Dr. Exner with patients at the hospital.
  Fact 0.82: Chris is devastated that Liza left voluntarily to help at the hospital.
  Fact 0.83: Travis promises to bring Liza back.
  Fact 0.84: Madison discovers Daniel detaining Adams in the Trans' basement.
  Fact 0.85: Alicia and Chris get drunk in and vandalize the abandoned home of a wealthy family.
  Fact 0.86: Strand recruits Nick for an escape plot.
  Fact 0.87: Travis convinces Moyers' squad to take him to the hospital to check on his friends.
  Fact 0.88: While en route, Moyers encourages Travis to shoot a zombie.
  Fact 0.89: Travis is emotionally unable to pull the trigger.
  Fact 0.90: The soldiers stop to assist another squad in a building infested by zombies.
  Fact 0.91: Most of those soldiers, including Moyers, are overcome.
  Fact 0.92: The few survivors flee and drop off Travis near the Safe Zone.
  Fact 0.93: Travis learns that Daniel tortured Adams into revealing what 'Cobalt' means.
  Fact 0.94: 'Cobalt' means that in the morning, all civilians will be killed, and the guardsmen will evacuate the city.
  Fact 0.95: Griselda dies of septic shock at the hospital.
  Fact 0.96: Liza shoots Griselda's brain to prevent reanimation.
  Fact 0.97: Daniel visits a nearby sports arena to verify Adams' story that it was sealed with 2,000 now-zombified civilians inside.
  Fact 0.98: Episode 'The Good Man' is overall number 6 and season number 6.
  Fact 0.99: Episode 'The Good Man' was directed by Stefan Schwartz and written by Robert Kirkman and Dave Erickson.
  Fact 0.100: Episode 'The Good Man' originally aired on October 4, 2015.
  Fact 0.101: Episode 'The Good Man' had 6.86 million U.S. viewers.
  Fact 0.102: The group drives to the National Guard's headquarters to rescue Liza, Griselda, and Nick.
  Fact 0.103: Adams agrees to be their guide when let go by Travis.
  Fact 0.104: The group infiltrates the base after Daniel distracts the guards by leading a horde of walkers from the arena.
  Fact 0.105: Travis, Madison, Daniel, and Ofelia go inside, while Alicia and Chris stay behind.
  Fact 0.106: Meanwhile, the walkers breach the perimeter defenses and swarm the base.
  Fact 0.107: Travis's group reaches the holding cells and sets the detainees free.
  Fact 0.108: Travis's group reunites with Nick, Liza, and Strand before trying to escape.
  Fact 0.109: The group tries to escape through the medical ward.
  Fact 0.110: The group discovers Dr. Exner has euthanized all of the patients in the medical ward.
  Fact 0.111: Dr. Exner tells them of an escape route before presumably committing suicide.
  Fact 0.112: Before they can escape, the group encounters Adams, who shoots Ofelia in the arm.
  Fact 0.113: Enraged, Travis brutally beats Adams and leaves him for dead.
  Fact 0.114: Strand leads the group to his oceanside mansion.
  Fact 0.115: At the mansion, Strand reveals to Nick that he owns a yacht which he plans to escape on, called the Abigail.
  Fact 0.116: On the beach, Liza reveals to Madison that she had been bitten during the escape.
  Fact 0.117: Liza pleads with Madison and Travis to euthanize her before she turns.
  Fact 0.118: Travis promises to protect Chris before shooting Liza.

Retrieved items:
<item 1>
Fear the Walking Dead (season 1)
Chris ' bus is trapped in a traffic jam caused by a zombie shooting by the police . He films the event and joins in a protest against that and other recent
fatal police shootings , when Travis and Liza meet up with him . A riot erupts after police shoot down another zombie , but the three Manawas find refuge with
the Salazars in their gated barbershop - home . Travis tells Madison to take the kids to the desert without him ; he will catch up . Alicia witnesses her
zombified neighbor , Mr. Dawson , attacking Mrs. Cruz across the street , but Madison prevents her from intervening . The group inside the barbershop remains
trapped , while the riot outside intensifies . `` The Dog '' Adam Davidson Jack LoGiudice September 13 , 2015 ( 2015 - 09 - 13 ) 7.19 While a riot rages outside
, a mob sets fire to the store adjoining the barbershop , forcing the Salazars and Manawas to flee . The group reaches Travis ' truck and escapes , but not
before Griselda is injured by a collapsing scaffold .
</item 1>

<item 2>
Fear the Walking Dead (season 1)
`` Not Fade Away '' Kari Skogland Meaghan Oppenheimer September 20 , 2015 ( 2015 - 09 - 20 ) 6.62 Days after the National Guard quarantines the neighborhood
into a Safe Zone , residents try to live normally . Tensions build under the military rule . Madison is stressed by extra work caused by her home 's being
overcrowded and Travis ' new role as civilian liaison . Chris shows a video to Travis and Madison of a light signaling from the Dead Zone . Travis talks Doug
into getting mental help . Liza medically helps neighbors . Nick steals morphine from Hector via IV drip . Ofelia kisses Adams , who was unable to get Griselda
's medicine . Madison exchanges signals with the flashing light in the Dead Zone . She sneaks outside the fence to investigate and finds evidence that the
guardsmen killed civilians , even the uninfected . Travis learns that Doug has been hospitalized for his mental issues . Dr. Exner determines that Liza is not
technically a nurse .
</item 2>

<item 3>
*Fear the Walking Dead* season 1, AMC’s six-episode prequel to *The Walking Dead*, aired August 23–October 4, 2015. It starred Kim Dickens, Cliff Curtis, Frank
Dillane, Alycia Debnam-Carey, Elizabeth Rodriguez, Mercedes Mason, Lorenzo James Henrie, and Rubén Blades. Reviews were generally favorable, though some
criticized its slow pacing; Metacritic scored it 66/100. The premiere drew
</item 3>

<item 4>
Fear the Walking Dead (season 1)
6 6 `` The Good Man '' Stefan Schwartz Robert Kirkman & Dave Erickson October 4 , 2015 ( 2015 - 10 - 04 ) 6.86 The group drives to the National Guard 's
headquarters to rescue Liza , Griselda , and Nick . Adams agrees to be their guide when let go by Travis . The group infiltrates the base after Daniel distracts
the guards by leading a horde of walkers from the arena . Travis , Madison , Daniel , and Ofelia go inside , while Alicia and Chris stay behind . Meanwhile ,
the walkers breach the perimeter defenses and swarm the base . Travis ' group reach the holding cells and set the detainees free before reuniting with Nick ,
Liza , and Strand . They try to escape through the medical ward , where they discover Dr. Exner has euthanized all of the patients . Dr. Exner tells them of an
escape route before presumably committing suicide . Before they can escape , the group encounters Adams , who shoots Ofelia in the arm . Enraged , Travis
brutally beats Adams and leaves him for dead .
</item 4>

<item 5>
Fear the Walking Dead (season 1)
Alicia and Chris get drunk in and vandalize the abandoned home of a wealthy family . Strand recruits Nick for an escape plot . Travis convinces Moyers ' squad
to take him to the hospital to check on his friends . While en route , Moyers encourages Travis to shoot a zombie , but Travis is emotionally unable to pull the
trigger . The soldiers stop to assist another squad in a building infested by zombies , and most of those soldiers , including Moyers , are overcome . The few
survivors flee and drop off Travis near the Safe Zone . Travis learns that Daniel tortured Adams into revealing what `` Cobalt '' means : in the morning , all
civilians will be killed , and the guardsmen will evacuate the city . Griselda dies of septic shock at the hospital ; Liza shoots her brain to prevent
reanimation . Daniel visits a nearby sports arena to verify Adams ' story that it was sealed with 2,000 now - zombified civilians inside .
</item 5>

<item 6>
Fear the Walking Dead (season 1)
1 Talking Dead Fear the Walking Dead Episodes season 1 Web series The Walking Dead Torn Apart Cold Storage Fear the Walking Dead Flight 462 Miscellaneous Video
games Season One `` 400 Days '' Two Michonne Three : A New Frontier Other games Survival Instinct Road to Survival Rides The Walking Dead : The Ride Novels Rise
of the Governor The Road to Woodbury The Fall of the Governor Descent Invasion Search and Destroy Return to Woodbury Characters Lists Comic book series
Television series Aaron Alden Amy Jessie Anderson Andrea Axel Tara Chambler Clementine Daryl Dixon Merle Dixon Dwight Rosita Espinosa Lee Everett King Ezekiel
Abraham Ford Gareth The Governor Gregory Beth Greene Hershel Greene Maggie Greene Carl Grimes Lori Grimes Rick Grimes Dale Horvath Jadis Morgan Jones Lilly
Milton Mamet Caesar Martinez Deanna Monroe Paul `` Jesus '' Monroe Spencer Monroe Morales Michonne Negan Olivia The Prisoners Carol Peletier Sophia Peletier
Eugene Porter Glenn Rhee Lizzie and Mika Samuels Simon Gabriel Stokes Bob Stookey T - Dog Shane Walsh Sasha Williams Tyreese Williams Retrieved from ``
https://en.wikipedia.org/w/index.
</item 6>

<item 7>
Fear the Walking Dead (season 1)
overall No. in season Title Directed by Written by Original air date U.S. viewers ( millions ) `` Pilot '' Adam Davidson Robert Kirkman & Dave Erickson August
23 , 2015 ( 2015 - 08 - 23 ) 10.13 Nick awakens in a heroin den in an abandoned church , to find his girlfriend Gloria eating a corpse . While fleeing , he is
hit by a car and hospitalized . The doctor tells Madison and Travis that Nick 's claims about the incident are heroin hallucinations , but Travis believes Nick
after visiting the church himself . Alicia becomes more worried about Nick 's chemical dependency . The next day , school closes early due to the high levels of
absenteeism and rumors of an epidemic . Nick escapes from the hospital and meets with Calvin , hoping to learn if the drugs Calvin sold him caused him to
hallucinate in the church . Calvin tries to kill Nick to prevent him from exposing Calvin as a drug dealer or impugning the quality of his drugs . In the
ensuing struggle , Calvin is mortally shot . After Travis and Madison arrive , the zombified Calvin attacks them .
</item 7>

<item 8>
Fear the Walking Dead (season 1)
Rotten Tomatoes . Retrieved August 25 , 2015 . Jump up ^ `` Fear the Walking Dead : Season 1 '' . Metacritic . Retrieved August 25 , 2015 . Jump up ^
Vincentelli , Elisabeth ( August 20 , 2015 ) . `` ' Walking Dead ' prequel is creepily suspenseful '' . New York Post . Retrieved August 20 , 2015 . Jump up ^
Tucker , Ken ( August 5 , 2015 ) . `` ' Fear the Walking Dead ' : Artful Suspense , Not Bloody Horror '' . Yahoo TV . Retrieved August 20 , 2015 . Jump up ^
Goodman , Tim ( August 4 , 2015 ) . `` ' Fear the Walking Dead ' : TV Review '' . The Hollywood Reporter . Retrieved August 20 , 2015 . Jump up ^ Fienberg ,
Daniel ( August 20 , 2015 ) . `` No. 296 - ' Fear the Walking Dead ' and more '' . HitFix .
</item 8>

<item 9>
Fear the Walking Dead (season 1)
Fear the Walking Dead ( season 1 ) - wikipedia Fear the Walking Dead ( season 1 ) Jump to : navigation , search Fear the Walking Dead ( season 1 ) Home media
cover art Starring Kim Dickens Cliff Curtis Frank Dillane Alycia Debnam - Carey Elizabeth Rodriguez Mercedes Mason Lorenzo James Henrie Rubén Blades Country of
origin United States No. of episodes 6 Release Original network AMC Original release August 23 ( 2015 - 08 - 23 ) -- October 4 , 2015 ( 2015 - 10 - 04 ) Season
chronology Next → Season 2 List of Fear the Walking Dead episodes The first season of Fear the Walking Dead , an American horror - drama television series on
AMC , premiered on August 23 , 2015 , and concluded on October 4 , 2015 , consisting of six episodes . The series is a companion series and prequel to The
Walking Dead , which is based on the comic book series of the same name by Robert Kirkman , Tony Moore , and Charlie Adlard .
</item 9>

<item 10>
Fear the Walking Dead (season 1)
TV by the Numbers . October 2 , 2015 . Retrieved March 27 , 2018 . Jump up ^ Lambert , David ( October 6 , 2015 ) . `` Fear the Walking Dead - Press Release
Announces ' The Complete 1st Season ' on Blu - ray , DVD '' . TVShowsOnDVD.com . Retrieved October 6 , 2015 . Jump up ^ Lambert , David ( January 20 , 2016 ) .
`` Fear the Walking Dead - ' The Complete 1st Season : Special Edition 's ' Press Release ! '' . TVShowsOnDVD.com . Retrieved January 21 , 2016 . External links
( edit ) Official website List of Fear the Walking Dead episodes on IMDb ( hide ) The Walking Dead Robert Kirkman Tony Moore Charlie Adlard Comic books The
Walking Dead Television series The Walking Dead Episodes season 1 5 6 7 8 Accolades Soundtrack Vol.
</item 10>
=== END ITEM id=95e2f4d14477bb3e ===

=== ITEM id=e1693134f5bff593 ===
Gold sentences and facts:
Sentence 0: Deadpool 2 is scheduled to be released in the United States on May 18 , 2018 . A sequel , Deadpool 3 , is in development .
  Fact 0.0: Deadpool 2 is scheduled to be released in the United States on May 18, 2018.
  Fact 0.1: A sequel, Deadpool 3, is in development.

Retrieved items:
<item 1>
Deadpool 2
Officially titling the film Deadpool 2 , the trailer also revealed that Crews was portraying Bedlam ; that Delaney was in the film , portraying Peter ; and that
Bill Skarsgård also had a role . The film 's final trailer , released a month later , confirmed that Lewis Tan would portray Shatterstar , joining Domino ,
Bedlam , and Peter as members of the X-Force in the film . Music Deadpool composer Junkie XL chose not to return for the sequel due to Miller 's departure . In
October 2017 , Tyler Bates was hired to write the score for the film . Release Deadpool 2 is set to be released in the United States on May 18 , 2018 . It was
previously scheduled for release on June 1 of that year . Marketing For the Fox presentation at CineEurope 2017 in June , Reynolds made a video message
featuring himself in costume as Deadpool from the film 's set . The first teaser poster , which pays homage to Norman Rockwell 's 1943 painting Freedom from
Want , was released that November .
</item 1>

<item 2>
Graph query row:
movie: Deadpool 3
release_date: None
related_work: Deadpool 2
relationship_type: RELATED_TO
connection: None
</item 2>

<item 3>
Deadpool 2
Deadpool 2 - wikipedia Deadpool 2 Jump to : navigation , search Deadpool 2 Teaser poster Directed by David Leitch Produced by Simon Kinberg Ryan Reynolds Lauren
Shuler Donner Written by Rhett Reese Paul Wernick Ryan Reynolds Based on Deadpool by Rob Liefeld Fabian Nicieza Starring Ryan Reynolds Josh Brolin Morena
Baccarin Julian Dennison Zazie Beetz T.J. Miller Brianna Hildebrand Jack Kesy Stefan Kapičić Music by Tyler Bates Cinematography Jonathan Sela Edited by Craig
Alpert Elísabet Ronaldsdóttir Production company Marvel Entertainment The Donners ' Company Genre Films Distributed by 20th Century Fox Release date May 18 ,
2018 ( 2018 - 05 - 18 ) ( United States ) Country United States Language English Deadpool 2 is an upcoming American superhero film based on the Marvel Comics
character Deadpool , distributed by 20th Century Fox . It is intended to be the eleventh installment in the X-Men film series , and a sequel to the 2016 film
Deadpool .
</item 3>

<item 4>
Deadpool 2
During filming , stunt woman Joi `` SJ '' Harris died in a motorcycle accident . Deadpool 2 is scheduled to be released in the United States on May 18 , 2018 .
A sequel , Deadpool 3 , is in development . Contents ( hide ) 1 Premise 2 Cast 3 Production 3.1 Development 3.2 Pre-production 3.3 Filming 3.4 Post-production 4
Music 5 Release 5.1 Marketing 6 Box office projection 7 Sequel 8 References 9 External links Premise Deadpool forms a team of mutants called the X-Force to
protect young mutant Russell from the time - traveling soldier Cable . Cast Ryan Reynolds as Wade Wilson / Deadpool : A wisecracking mercenary with accelerated
healing but severe scarring over his body after undergoing an experimental regenerative mutation . He forms the X-Force , a team of mutants . Josh Brolin as
Nathan Summers / Cable : A time travelling cybernetic mutant soldier , `` in many ways the opposite of Deadpool '' .
</item 4>

<item 5>
Deadpool 2
As part of a promotion for the film with beverage company Mike 's Harder , Sister Margaret 's School for Wayward Girls -- the fictional bar featured in the
films -- will be emulated in pop - ups at the Alligator Lounge in New York from April 26 -- 28 and the Slipper Clutch in Los Angeles from May 10 -- 12 . The pop
- ups will respectively serve pizza and chimichangas , traditionally Deadpool 's favorite food , as well as Deadpool - inspired Harder drinks . The campaign
also includes the chance to win a trip to the film 's premiere through Harder , and the Los Angeles pop - up will benefit the nonprofit DTLA Film Festival . Box
office projection Initial box office projections for the film in March 2018 expected it to gross $100 million over its opening weekend in the United States and
Canada , with a final domestic total expected to be lower than that of the first film ( $363 million ) at around $242 million . On April 20 , 2018 , both
Fandango and Regal Cinemas announced that Deadpool 2 was the best pre-selling R - rated film in their respective histories .
</item 5>

<item 6>
Deadpool 2
Josh Brolin emerged as a `` surprise contender '' to play Cable in April , ahead of Shannon and Harbour , and was officially cast in the role . Brolin also
portrays the Marvel Comics character Thanos , in the Marvel Cinematic Universe . Leslie Uggams confirmed that she would be reprising her role of Blind Al from
the first film , while Fox gave the sequel a June 1 , 2018 release date . Noting the release date , Leitch wanted to ensure that the film `` was worthy of a
summer tentpole movie , and we knew we were going to be wedged in between some big films '' , specifically wanting to expand the action and make the general
feeling of the film `` bigger '' than the first one . However , he wanted the film to have the same DNA as the original `` in terms of the tone , and the fun
... I love that challenge '' of combining that with the increased scope . The film uses Deadpool 's fourth wall - breaking to reference this release period ,
with Leitch calling these comments `` definitely fresh and timely when they come up . ''
</item 6>

<item 7>
Deadpool 2
In January 2018 , the film 's release was moved up to May 18 , 2018 . A month later , Terry Crews was revealed to have a role in the film , the character
Shatterstar was confirmed to be appearing , and the production returned to Vancouver for six days of reshoots under a new working title , Daisy . While some
reports emerged by mid-March claiming that the reshoots were due to poor audience responses during test screenings of the film , and consisted of sweeping
changes , the film was soon confirmed to be testing better than the original did , up to 98 out of 100 over three different tests ( compared to a maximum score
of 91 during tests of the first film ) . Because of this , the reshoots were adding more of the elements that audiences responded positively to , including
additional material featuring Cable and Domino . By the end of the month , two hours - worth of filming had taken place in Los Angeles for a then - secret cameo
appearance , and the film 's first full trailer was released .
</item 7>

<item 8>
Deadpool 2
Retrieved December 16 , 2017 . External links Official website Deadpool 2 on IMDb Deadpool Fabian Nicieza Rob Liefeld Supporting characters Agent X Avengers
Blind Al Bob , Agent of Hydra Sandi Brandenberg Cable Copycat Death Domino Emrys Killebrew Negasonic Teenage Warhead Outlaw Shiklah Siryn Spider - Man Weasel
Wolverine X-Men Antagonists Ajax Black Box Black Swan Black Talon Black Tom Cassidy Bullseye Doctor Bong Dracula Hit - Monkey Flag - Smasher Juggernaut Madcap
Sabretooth Slayback Taskmaster T - Ray Thanos Trapster Typhoid Mary Publications Cable & Deadpool Mrs. Deadpool and the Howling Commandos Wolverine and Deadpool
In other media Deadpool ( video game ) Deadpool ( film ) accolades box office records soundtrack No Good Deed Deadpool 2 Related articles 5 Ronin Gwenpool Mercs
for Money X-Force Deathstroke X-Men in other media Films X-Men X2 X-Men : The Last Stand soundtrack X-Men Origins : Wolverine X-Men : First Class The Wolverine
X-Men : Days of Future Past Deadpool soundtrack X-Men : Apocalypse Logan Deadpool 2 X-Men : Dark Phoenix The New Mutants Development of Gambit Characters
</item 8>

<item 9>
Deadpool 2
By April 26 , opening weekend industry projections had reached upwards of $150 million . Sequel By November 2016 , with development underway on Deadpool 2 , Fox
was also planning Deadpool 3 , which was said to include the team X-Force . With the confirmation that Leitch would direct Deadpool 2 , it was revealed that Fox
was looking for a separate filmmaker to direct Deadpool 3 . In March 2017 , Reese clarified that though Deadpool 2 sets - up the X-Force team , a future film
focused on the team would be separate from Deadpool 3 , `` so I think we 'll be able to take two paths . ( X-Force ) is where we 're launching something bigger
, but then ( Deadpool 3 is ) where we 're contracting and staying personal and small . '' After the proposed acquisition of 21st Century Fox by Disney was
announced in December 2017 , Disney CEO Bob Iger said that Deadpool would be integrated with the Marvel Cinematic Universe under Disney , and that the company
would be willing to make future R - rated Deadpool films `` as long as we let the audiences know what 's coming '' .
</item 9>

<item 10>
Deadpool 2
: Next Dimension X-Men Legends X-Men Legends II : Rise of Apocalypse X-Men : The Official Game X-Men : Destiny Wolverine Wolverine Wolverine : Adamantium Rage
X-Men : Wolverine 's Rage X2 : Wolverine 's Revenge X-Men Origins : Wolverine Characters in other media Apocalypse Cyclops Gambit Magneto Jean Grey Professor X
Psylocke Rogue Storm Wolverine See also X-Men storylines X-Men comics Live - action theatrical films based on Marvel Comics Franchises Ant - Man Ant - Man Ant -
Man and the Wasp Avengers The Avengers Avengers : Age of Ultron Avengers : Infinity War Untitled Avengers film production Blade Blade Blade II Blade : Trinity
Captain America Captain America ( serial ) Captain America ( 1990 ) Captain America : The First Avenger Captain America : The Winter Soldier Captain America :
Civil War Daredevil Daredevil Elektra Deadpool Deadpool Deadpool 2 Fantastic Four The Fantastic Four ( 1994 ) Fantastic Four ( 2005 ) Fantastic Four : Rise of
the Silver Surfer Fantastic Four ( 2015 ) Ghost Rider Ghost Rider Ghost Rider : Spirit of Vengeance Guardians of the Galaxy Guardians of the Galaxy Guardians of
the Galaxy Vol.
</item 10>
=== END ITEM id=e1693134f5bff593 ===

=== ITEM id=85947773d760dd2c ===
Gold sentences and facts:
Sentence 0: The Partridge Family Genre Musical sitcom Created by Bernard Slade Starring Shirley Jones David Cassidy Danny Bonaduce Susan Dey Suzanne Crough Jeremy Gelbwaks Dave Madden Brian Forster Theme music composer Diane Hilderbrand Danny Janssen Wes Farrell Opening theme `` When We 're Singin ' '' ( 1970 -- 71 ) `` C'mon , Get Happy '' ( 1971 -- 74 ) Composer ( s ) George Duning Benny Golson Warren Barker Hugo Montenegro Shorty Rogers Country of origin United States Original language ( s ) English No. of seasons No. of episodes 96 ( list of episodes ) Production Executive producer ( s ) Bob Claver Producer ( s ) William Bickley Paul Junger Witt Dale McRaven Larry Rosen Mel Swope Cinematography Fred Jackman , Jr . Irving Lippman Camera setup Single - camera Running time 25 minutes Production company ( s ) Screen Gems Television ( 1970 -- 1974 ) Distributor Columbia Pictures Television ( 1974 -- 84 ; 1989 -- 96 ) DFS Program Exchange ( 1984 -- 87 ) The Program Exchange ( 1987 -- 89 ) Columbia TriStar Television ( 1996 -- 2002 ) Sony Pictures Television ( 2002 -- present ) Release Original network ABC Audio format Monaural Original release September 25 , 1970 ( 1970 - 09 - 25 ) -- March 23 , 1974 ( 1974 - 03 - 23 ) Chronology Related shows Getting Together Goober and the Ghost Chasers Partridge Family 2200 A.D.
  Fact 0.0: The Partridge Family is a musical sitcom
  Fact 0.1: The Partridge Family was created by Bernard Slade
  Fact 0.2: The Partridge Family starred Shirley Jones
  Fact 0.3: The Partridge Family starred David Cassidy
  Fact 0.4: The Partridge Family starred Danny Bonaduce
  Fact 0.5: The Partridge Family starred Susan Dey
  Fact 0.6: The Partridge Family starred Suzanne Crough
  Fact 0.7: The Partridge Family starred Jeremy Gelbwaks
  Fact 0.8: The Partridge Family starred Dave Madden
  Fact 0.9: The Partridge Family starred Brian Forster
  Fact 0.10: The Partridge Family's theme music composers were Diane Hilderbrand, Danny Janssen and Wes Farrell
  Fact 0.11: The Partridge Family's opening theme was 'When We're Singin'' (1970-71)
  Fact 0.12: The Partridge Family's opening theme was 'C'mon, Get Happy' (1971-74)
  Fact 0.13: The Partridge Family's composers were George Duning, Benny Golson, Warren Barker, Hugo Montenegro and Shorty Rogers
  Fact 0.14: The Partridge Family is from the United States
  Fact 0.15: The Partridge Family's original language is English
  Fact 0.16: The Partridge Family has 96 episodes
  Fact 0.17: The Partridge Family's executive producer was Bob Claver
  Fact 0.18: The Partridge Family's producers were William Bickley, Paul Junger Witt, Dale McRaven, Larry Rosen and Mel Swope
  Fact 0.19: The Partridge Family's cinematography was by Fred Jackman, Jr. and Irving Lippman
  Fact 0.20: The Partridge Family uses a single-camera setup
  Fact 0.21: The Partridge Family has a running time of 25 minutes
  Fact 0.22: The Partridge Family was produced by Screen Gems Television (1970-1974)
  Fact 0.23: The Partridge Family was distributed by Columbia Pictures Television (1974-84; 1989-96)
  Fact 0.24: The Partridge Family was distributed by DFS Program Exchange (1984-87)
  Fact 0.25: The Partridge Family was distributed by The Program Exchange (1987-89)
  Fact 0.26: The Partridge Family was distributed by Columbia TriStar Television (1996-2002)
  Fact 0.27: The Partridge Family was distributed by Sony Pictures Television (2002-present)
  Fact 0.28: The Partridge Family aired originally on ABC
  Fact 0.29: The Partridge Family has a monaural audio format
  Fact 0.30: The Partridge Family originally aired from September 25, 1970 to March 23, 1974
  Fact 0.31: Related shows include Getting Together
  Fact 0.32: Related shows include Goober and the Ghost Chasers
  Fact 0.33: Related shows include Partridge Family 2200 A.D.

Retrieved items:
<item 1>
The Partridge Family
Actor David Cassidy sang lead on most of the records , and actor Shirley Jones can be heard on a few . But the other voices and all the instruments were
recorded by session musicians . The Partridge Family 's instrumental sound was provided by musicians informally known as The Wrecking Crew , who did many of the
pop hits of the day attributed to other performers . The actors pretended to play their instruments . None of the actors sang background ; they lip - synched to
recorded singing by The Love Generation , AKA The Ron Hicklin Singers , composed of brothers John and Tom Bahler , Jackie Ward ( who as Robin Ward had a # 14
hit `` Wonderful Summer '' on Dot Records in 1963 ) , and Ron Hicklin . Actor David Cassidy was originally to lip sync with the rest of the cast , but convinced
Farrell that he could sing and was allowed to join the studio ensemble as the lead singer . He and his step - mother , Shirley , are the only two show actors
allowed to sing for recording . The Partridge Family theme song , produced by Rogers , was shown over opening credits . More than one song was used as the theme
song .
</item 1>

<item 2>
The Partridge Family
The company promoted the show by releasing a series of albums featuring the family band , though David Cassidy and Shirley Jones , who sang background , were
the only cast members who were actually featured on the recordings . As the show and other associated merchandising soared , David Cassidy became a teen idol .
The producers signed Cassidy as a solo act as well . Cassidy began touring with his own group of musicians , performing Partridge songs as well as hits from his
own albums , to thousands of screaming teenagers in major stadiums across the USA , UK , Europe , Japan and Australia . The Partridge Family 's biggest hit came
in 1970 with the song `` I Think I Love You '' , written by Tony Romeo ( who had previously written several of the Cowsills ' hits ) , peaked at Number 1 on the
Billboard charts in December of that year . It sold over five million copies , outselling the Beatles ' `` Let It Be '' , was awarded a gold disc , and made the
group the third fictional artist to have a No. 1 hit ( after The Chipmunks and The Archies ) .
</item 2>

<item 3>
The Partridge Family
The shows often contrast suburban life with the adventures of a show - business family on the road . After the first season , more of the show 's action takes
place in their hometown than on tour . History ( edit ) This section needs additional citations for verification . Please help improve this article by adding
citations to reliable sources . Unsourced material may be challenged and removed . ( May 2016 ) ( Learn how and when to remove this template message ) Created
for television by Bernard Slade , the series ' executive producer was Bob Claver . The show was inspired by and loosely based on The Cowsills , a family pop
music group that was famous in the late 1960s . In the show 's early development , the Cowsill children were considered by the producers , but because the
Cowsills were not trained actors and were too old for the roles as scripted , Slade and Claver abandoned that idea . Shirley Jones had already been signed as
mother Shirley Partridge and star of the show . The pilot was filmed in December 1969 . This unaired pilot differs from the pilot broadcast in 1970 .
</item 3>

<item 4>
Graph query row:
person: Tony Romeo
hit: I Think I Love You
hitRelationship: CREATED
hitRole: None
relatedTo: The Partridge Family
connection: ['CREATED', 'RELATED_TO']
</item 4>

<item 5>
The Partridge Family
1 hit ( after The Chipmunks and The Archies ) . The song 's companion LP , The Partridge Family Album , reached Number 4 in the Billboard 200 . It was also
awarded a gold disc by the RIAA in December 1970 , having sold over 500,000 copies . A string of hit Partridge singles followed : `` Does n't Somebody Want to
Be Wanted '' , `` I 'll Meet You Halfway '' , `` I Woke Up In Love This Morning '' , `` It 's One of Those Nights ( Yes Love ) '' , `` Am I Losing You '' , ``
Looking Through the Eyes of Love '' , `` Breaking Up Is Hard to Do '' , and `` A Friend and a Lover '' . These singles were showcased on million - selling
albums including Up To Date , Sound Magazine , Shopping Bag , Notebook , Crossword Puzzle , and Bulletin Board . Their holiday album A Partridge Family
Christmas Card was the No. 1 selling Christmas record of 1971 .
</item 5>

<item 6>
The Partridge Family
( hide ) The Partridge Family David Cassidy Shirley Jones Danny Bonaduce Susan Dey Jeremy Gelbwaks Brian Forster Suzanne Crough Dave Madden Studio albums The
Partridge Family Album Up to Date Sound Magazine A Partridge Family Christmas Card Shopping Bag The Partridge Family Notebook Crossword Puzzle Bulletin Board
Other albums At Home with Their Greatest Hits The World of the Partridge Family Greatest Hits The Definitive Collection Come On Get Happy ! : The Very Best of
The Partridge Family Singles `` I Think I Love You '' `` Does n't Somebody Want to Be Wanted '' `` I 'll Meet You Halfway '' `` I Woke Up in Love This Morning
'' `` It 's One of Those Nights ( Yes Love ) '' `` Breaking Up Is Hard to Do '' `` Looking Through the Eyes of Love '' Related articles List of The Partridge
Family episodes The Partridge Family discography David Cassidy discography Wes Farrell Goober and the Ghost Chasers Partridge Family 2200 A.D. MusicBrainz :
3376b487 - 95b4 - 42e6 - b7f4 - 71cffffcad97 Retrieved from `` https://en.wikipedia.org/w/index.
</item 6>

<item 7>
The Partridge Family
The Partridge Family - wikipedia The Partridge Family Jump to : navigation , search The Partridge Family Genre Musical sitcom Created by Bernard Slade Starring
Shirley Jones David Cassidy Danny Bonaduce Susan Dey Suzanne Crough Jeremy Gelbwaks Dave Madden Brian Forster Theme music composer Diane Hilderbrand Danny
Janssen Wes Farrell Opening theme `` When We 're Singin ' '' ( 1970 -- 71 ) `` C'mon , Get Happy '' ( 1971 -- 74 ) Composer ( s ) George Duning Benny Golson
Warren Barker Hugo Montenegro Shorty Rogers Country of origin United States Original language ( s ) English No. of seasons No. of episodes 96 ( list of episodes
) Production Executive producer ( s ) Bob Claver Producer ( s ) William Bickley Paul Junger Witt Dale McRaven Larry Rosen Mel Swope Cinematography Fred Jackman
, Jr .
</item 7>

<item 8>
The Partridge Family
wikipedia.org/w/index.php?title=The_Partridge_Family&oldid=837155910 '' Categories : The Partridge Family 1970 American television series debuts 1974 American
television series endings 1970s American sitcoms American Broadcasting Company network shows American television sitcoms Bell Records artists English - language
television programs Fictional families Fictional musical groups Musical television series Television series based on singers and musicians Television series by
Sony Pictures Television Television shows set in California Television series about families Television programs adapted into comics Hidden categories : Use mdy
dates from October 2012 Articles needing additional references from May 2016 All articles needing additional references Articles containing potentially dated
statements from January 2011 All articles containing potentially dated statements Wikipedia articles with MusicBrainz identifiers Talk Contents About Wikipedia
Wikiquote Deutsch Español فارسی Français Italiano Nederlands 日本 語 Norsk nynorsk Português Русский Simple English Srpskohrvatski / српскохрватски Svenska 4 more
Edit links This page was last edited on 19 April 2018 , at 02 : 06 .
</item 8>

<item 9>
The Partridge Family
More than one song was used as the theme song . Season 1 episodes feature the song `` When We 're Singin ' '' , which was replaced in subsequent seasons with ``
C'mon Get Happy '' . `` Five of us , and Mom working all day , we knew we could help her if our music would pay , Danny got Reuben to sell our song , and it
really came together when Mom sang along ... '' ( from `` When We 're Singin ' '' ) Later , under Farrell 's direction , the new version had new lyrics sung to
the `` When We 're Singin ' '' tune . With the new chorus finalized , `` C'mon Get Happy '' showcased the new verse : `` We had a dream , we 'd go travelin '
together , We 'd spread a little lovin ' then we 'd keep movin ' on .
</item 9>

<item 10>
The Partridge Family
Jump up ^ `` TSORT Song Artist 592 -- The Partridge Family '' . TSORT.info . Retrieved June 22 , 2010 . Jump up ^ `` TSORT Album Artist 994 -- The Partridge
Family '' . TSORT.info . Retrieved June 22 , 2010 . Jump up ^ Whitburn , Joel ( 2004 ) . Christmas in the Charts ( 1920 -- 2004 ) . Wisconsin : Record Research
Inc. p. 179 . ISBN 0 - 89820 - 161 - 6 . Jump up ^ Parry , Wayne ( April 10 , 2011 ) . David Cassidy , Danny Bonaduce play Partridge song . Associated Press .
Retrieved 2011 - 04 - 10 . Jump up ^ Coopee , Todd . `` The Partridge Family Game '' . ToyTales.ca . Jump up ^ Shirley , Ian ( 2005 ) . Can Rock & Roll Save the
World ? : An Illustrated History of Music and Comics . SAF Publishing Ltd. p. 88 - 89 .
</item 10>
=== END ITEM id=85947773d760dd2c ===

=== ITEM id=6872d92d5d9aa9e4 ===
Gold sentences and facts:
Sentence 0: `` The Glory of Love '' is a song written by Billy Hill , recorded by Benny Goodman in 1936 , whose version was a number one pop hit . In 1951 , R&B vocal group , The Five Keys , had their biggest R&B hit with their version of the song , hitting number one on the R&B chart for four non-consecutive weeks . Although The Five Keys recording sold a reported million copies pressed recordings are very rare .
  Fact 0.0: The Glory of Love is a song.
  Fact 0.1: The Glory of Love was written by Billy Hill.
  Fact 0.2: The Glory of Love was recorded by Benny Goodman in 1936.
  Fact 0.3: Benny Goodman's version of The Glory of Love was a number one pop hit.
  Fact 0.4: In 1951, the R&B vocal group The Five Keys had their biggest R&B hit with their version of The Glory of Love.
  Fact 0.5: The Five Keys' version hit number one on the R&B chart for four non-consecutive weeks.
  Fact 0.6: The Five Keys' recording reportedly sold a million copies.
  Fact 0.7: Pressed recordings of The Five Keys' version are very rare.
Sentence 1: `` The Glory of Love '' Single by Benny Goodman Released 1936 Format 78 rpm vinyl Recorded 1936 Genre Pop Songwriter ( s ) Billy Hill Benny Goodman singles chronology `` The Glory of Love ' '' ( 1936 ) The Glory of Love'1936 The Five Keys singles chronology `` The Glory of Love ' '' ( 1951 ) The Glory of Love'1951 `` Yes Sir , That 's My Baby '' ( 1952 ) Yes Sir , That 's My Baby1952
  Fact 1.0: The Glory of Love is a single by Benny Goodman.
  Fact 1.1: The Glory of Love was released in 1936.
  Fact 1.2: The Glory of Love has the format 78 rpm vinyl.
  Fact 1.3: The Glory of Love was recorded in 1936.
  Fact 1.4: The Glory of Love has the genre pop.
  Fact 1.5: The songwriter of The Glory of Love is Billy Hill.
  Fact 1.6: The Benny Goodman singles chronology lists The Glory of Love (1936).
  Fact 1.7: The Five Keys singles chronology lists The Glory of Love (1951).
  Fact 1.8: The Five Keys singles chronology lists Yes Sir, That's My Baby (1952).

Retrieved items:
<item 1>
Joy to the World (Three Dog Night song)
Sugar '' -- The Archies 1970 : `` Bridge over Troubled Water '' -- Simon & Garfunkel 1971 : `` Joy to the World '' -- Three Dog Night 1972 : `` The First Time
Ever I Saw Your Face '' -- Roberta Flack 1973 : `` Tie a Yellow Ribbon Round the Ole Oak Tree '' -- Tony Orlando and Dawn 1974 : `` The Way We Were '' -- Barbra
Streisand 1975 : `` Love Will Keep Us Together '' -- Captain & Tennille 1976 : `` Silly Love Songs '' -- Wings 1977 : `` Tonight 's the Night ( Gonna Be Alright
) '' -- Rod Stewart 1978 : `` Shadow Dancing '' -- Andy Gibb 1979 : `` My Sharona '' -- The Knack Complete list ( 1946 -- 1959 ) ( 1960 -- 1979 ) ( 1980 -- 1999
) ( 2000 -- 2019 ) Retrieved from `` https://en.wikipedia.org/w/index.
</item 1>

<item 2>
The Glory of Love (song)
Although The Five Keys recording sold a reported million copies pressed recordings are very rare . Contents ( hide ) 1 Charts 1.1 Weekly charts 2 Otis Redding
version 3 Charts 3.1 Weekly charts 4 References Charts ( edit ) Weekly charts ( edit ) Chart ( 1951 ) Peak position US Billboard Hot Rhythm & Blues Songs Otis
Redding version ( edit ) `` The Glory of Love '' Single by Otis Redding from the album The Dock of the Bay B - side `` I 'm Coming Home '' Released 1967 Format
7 '' vinyl Recorded December , 1967 Genre Soul , Pop Length 2 : 38 Label Volt S419 Songwriter ( s ) Billy Hill Producer ( s ) Steve Cropper Otis Redding singles
chronology `` Shake '' ( 1967 ) `` The Glory of Love '' ( 1967 ) `` Tramp '' ( 1967 ) `` Shake '' ( 1967 ) `` The Glory of Love '' ( 1967 ) `` Tramp '' ( 1967 )
In 1967 , Otis Redding recorded a cover version for his 1968 album , The Dock of the Bay .
</item 2>

<item 3>
The Glory of Love (song)
In 1951 , R&B vocal group , The Five Keys , had their biggest R&B hit with their version of the song , hitting number one on the R&B chart for four
non-consecutive weeks . Although The Five Keys recording sold a reported million copies pressed recordings are very rare .
</item 3>

<item 4>
Let My Love Open the Door
Billboard Hot 100 59 U.S. Cashbox 92 Personnel ( edit ) Pete Townshend : vocals , guitars , synths John `` Rabbit '' Bundrick : `` straight '' keyboards Simon
Phillips : drums Tony Butler : bass guitar Ted Jensen - mastering engineer Covers ( edit ) A version of this song by Luminate appears on The Covers EP . It was
also covered by M. Ward ( for the 2005 Starbucks compilation album , Sweetheart 2005 : Love Songs ) Christian rock group Audio Adrenaline covered this song on
their 1999 album Underdog . Natalie Imbruglia covered the song on her album Male released in 2015 . Great Big Sea covered the song on their 2012 album XX .
Roger Daltrey covered the song in 2015 for a campaign involving Teen Cancer America and First Citizens Bank . A full version was released in February 2016 ,
with all proceeds going to Teen Cancer America . The group Rogue Wave also covered it in `` Cover me '' released in 2017 .
</item 4>

<item 5>
Let My Love Open the Door
Initially , Townshend 's manager despised the track due to it `` not sounding like Townshend , '' and wanted it to be removed from Empty Glass . However , upon
the song 's chart success , his manager called to apologize . Despite the song 's critical and commercial success , Pete Townshend did not consider it one of
his best songs . He told Rolling Stone in an interview that `` Let My Love Open the Door '' was `` just a ditty , '' also claiming that he preferred his minor
US hit `` A Little Is Enough '' from the same album . In 1996 , Townshend released a new version of the song , called `` the E. Cola mix '' , turning the song
into a ballad . This version appeared in different television shows and film soundtracks . Chart performance ( edit ) Weekly charts ( edit ) Chart ( 1980 ) Peak
position Canada RPM 100 5 UK Singles Chart 46 U.S. Billboard Hot 100 9 U.S. Cash Box Top 100 11 Year - end charts ( edit ) Chart ( 1980 ) Rank Canada 35 U.S.
</item 5>

<item 6>
The Glory of Love (song)
Billboard . Retrieved 2012 - 11 - 22 . Jump up ^ `` Billboard R&B Singles Chart '' . Billboard . Retrieved 2016 - 01 - 27 . Preceded by `` Sixty Minute Man ''
by The Dominoes Billboard Best Selling Retail Rhythm & Blues Records number - one single ( The Five Keys version ) September 22 , 1951 Succeeded by `` Sixty
Minute Man '' by The Dominoes Retrieved from `` https://en.wikipedia.org/w/index.php?title=The_Glory_of_Love_(song)&oldid=814898703 '' Categories : 1936 songs
1951 singles Benny Goodman songs Otis Redding songs Songs written by Billy Hill ( songwriter ) Billboard Hot R&B / Hip - Hop Songs number - one singles Paul
McCartney songs Hidden categories : Articles needing additional references from December 2015 All articles needing additional references Music infoboxes with
deprecated parameters Articles with hAudio microformats Talk Contents About Wikipedia Norsk Edit links This page was last edited on 11 December 2017 , at 15 :
51 . About Wikipedia
</item 6>

<item 7>
The Glory of Love (song)
The Glory of Love ( song ) - wikipedia The Glory of Love ( song ) Jump to : navigation , search This article needs additional citations for verification .
Please help improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( December 2015 ) ( Learn how and
when to remove this template message ) For the song by Peter Cetera , see Glory of Love . `` The Glory of Love '' Single by Benny Goodman Released 1936 Format
78 rpm vinyl Recorded 1936 Genre Pop Songwriter ( s ) Billy Hill Benny Goodman singles chronology `` The Glory of Love ' '' ( 1936 ) The Glory of Love'1936 The
Five Keys singles chronology `` The Glory of Love ' '' ( 1951 ) The Glory of Love'1951 `` Yes Sir , That 's My Baby '' ( 1952 ) Yes Sir , That 's My Baby1952 ``
The Glory of Love '' is a song written by Billy Hill , recorded by Benny Goodman in 1936 , whose version was a number one pop hit .
</item 7>

<item 8>
The Glory of Love (song)
Redding 's cover became a top 20 hit and reached number nineteen on the Billboard R&B Songs chart and number sixty on the Billboard Hot 100 . Charts ( edit )
Weekly charts ( edit ) Chart ( 1967 ) Peak position US Billboard Hot 100 60 US Billboard Hot Rhythm & Blues Songs 19 References ( edit ) Jump up ^ Whitburn ,
Joel ( 2004 ) . Top R&B / Hip - Hop Singles : 1942 - 2004 . Record Research . p. 205 . Jump up ^ Propes , Steve ( 1973 ) . Those Oldies But Goodies : A Guide to
50 's Record Collecting . The Macmillan Company , New York . p. 42 . Jump up ^ Whitburn , Joel ( 2004 ) . Top R&B / Hip - Hop Singles : 1942 - 2004 . Record
Research . p. 486 . Jump up ^ `` allmusic ( ( ( Otis Redding > Awards ) ) ) '' . Billboard . Retrieved 2012 - 11 - 22 .
</item 8>

<item 9>
Graph query row:
songwriter: Billy Hill
song: The Glory of Love
relationship: CREATED
</item 9>

<item 10>
Let My Love Open the Door
Imdb.com. 2017 - 05 - 12 . Retrieved 2018 - 03 - 30 . External links ( edit ) Lyrics of this song at MetroLyrics ( hide ) Pete Townshend Discography Studio
albums Who Came First Rough Mix ( w / Ronnie Lane ) Empty Glass All the Best Cowboys Have Chinese Eyes White City : A Novel The Iron Man : A Musical
Psychoderelict Classic Quadrophenia Meher Baba tribute albums ( w / Ronnie Lane et al . ) Happy Birthday I Am With Love O ' Parvardigar Live albums Deep End
Live !
</item 10>
=== END ITEM id=6872d92d5d9aa9e4 ===

=== ITEM id=385905ae2cfbe32f ===
Gold sentences and facts:
Sentence 0: Mount Baker - Snoqualmie National Forest and Nooksack Falls in the North Cascades range of , Washington , deer hunting scenes .
  Fact 0.0: Mount Baker-Snoqualmie National Forest is in the North Cascades range of Washington
  Fact 0.1: Nooksack Falls is in the North Cascades range of Washington
  Fact 0.2: Deer hunting scenes are associated with Mount Baker-Snoqualmie National Forest
  Fact 0.3: Deer hunting scenes are associated with Nooksack Falls
Sentence 1: St. Theodosius Russian Orthodox Cathedral , in the Tremont neighborhood of Cleveland , Ohio . The name plaque is clearly visible in one scene . Lemko Hall , Cleveland , Ohio . Also located in Tremont , the wedding banquet was filmed here . The name is clearly visible in one scene . U.S. Steel Central Furnaces in Cleveland , Ohio . Opening sequence steel mill scenes . Patpong , Bangkok , Thailand , the area used to represent Saigon 's red light district . Sai Yok , Kanchanaburi Province , Thailand Mount Baker - Snoqualmie National Forest and Nooksack Falls in the North Cascades range of , Washington , deer hunting scenes . Steubenville , Ohio , for some mill and neighborhood shots . Struthers , Ohio , for external house and long - range road shots . Also including , the town 's bowling alley is the Bowladrome Lanes , located at 56 State Street , Struthers , Ohio . Weirton , West Virginia , for mill and trailer shots . River Kwai , Thailand , Prison camp and initial Russian roulette scene .
  Fact 1.0: St. Theodosius Russian Orthodox Cathedral is in the Tremont neighborhood of Cleveland, Ohio
  Fact 1.1: The name plaque of St. Theodosius Russian Orthodox Cathedral is clearly visible in one scene
  Fact 1.2: Lemko Hall is in Cleveland, Ohio
  Fact 1.3: Lemko Hall is also located in Tremont
  Fact 1.4: The wedding banquet was filmed at Lemko Hall
  Fact 1.5: The name of Lemko Hall is clearly visible in one scene
  Fact 1.6: U.S. Steel Central Furnaces are in Cleveland, Ohio
  Fact 1.7: Opening sequence steel mill scenes are at U.S. Steel Central Furnaces
  Fact 1.8: Patpong, Bangkok, Thailand is the area used to represent Saigon's red light district
  Fact 1.9: Sai Yok is a location in Kanchanaburi Province, Thailand
  Fact 1.10: Mount Baker-Snoqualmie National Forest is in the North Cascades range of Washington
  Fact 1.11: Nooksack Falls is in the North Cascades range of Washington
  Fact 1.12: Deer hunting scenes are associated with Mount Baker-Snoqualmie National Forest
  Fact 1.13: Deer hunting scenes are associated with Nooksack Falls
  Fact 1.14: Steubenville, Ohio is a location used for some mill and neighborhood shots
  Fact 1.15: Struthers, Ohio is a location used for external house and long-range road shots
  Fact 1.16: The town's bowling alley is the Bowladrome Lanes
  Fact 1.17: The Bowladrome Lanes are located at 56 State Street, Struthers, Ohio
  Fact 1.18: Weirton, West Virginia is a location used for mill and trailer shots
  Fact 1.19: River Kwai, Thailand is the location of the prison camp and initial Russian roulette scene

Retrieved items:
<item 1>
The Deer Hunter
The reception scene was filmed at nearby Lemko Hall . The amateur extras lined up for the crowded wedding - dance sequences drank real liquor and beer . The
scenes were filmed in the summer , but were set in the fall . To accomplish a look of fall , individual leaves were removed from deciduous trees . Zsigmond also
had to desaturate the colors of the exterior shots , partly in camera and in the laboratory processing . The production manager asked each of the Russian
immigrant extras to bring to the location a gift - wrapped box to double for wedding presents . The manager figured if the extras did this , not only would the
production save time and money , but the gifts would also look more authentic . Once the unit wrapped and the extras disappeared , the crew discovered to their
amusement that the boxes were n't empty but filled with real presents , from china to silverware . `` Who got to keep all these wonderful offerings , '' wrote
Deeley `` is a mystery I never quite fathomed . '' Cimino originally claimed that the wedding scene would take up 21 minutes of screen time . In the end , it
took 51 minutes .
</item 1>

<item 2>
The Deer Hunter
( Television Production ) . Chicago , IL : WTTW . ^ Jump up to : The Deer Hunter ( 1978 ) -- Filming locations Jump up ^ `` Lemko Hall '' . The Encyclopedia of
Cleveland History . May 24 , 2002 . Jump up ^ `` 7 Movie Locations in Bangkok - Bangkok.com Magazine '' . Bangkok.com . Retrieved 2016 - 10 - 17 . Jump up ^ Nam
, Suzanne ( 2012 ) . Moon Thailand . Avalon Travel . p. 320 . ISBN 9781598809695 . Jump up ^ Krause , Mariella ( 2013 ) . Lonely Planet Pacific Northwest 's
Best Trips . Lonely Planet . p. 118 . ISBN 1741798159 . Jump up ^ `` The Deer Hunter film locations ( 1978 ) '' . Movie-locations.com . Retrieved 2016 - 10 - 17
. Jump up ^ Randy Fox Writer and photographer ( 2012 - 05 - 05 ) .
</item 2>

<item 3>
The Deer Hunter
`` ' The Deer Hunter ' And ' Reckless ' : Hollywood Briefly Comes To The Factory Valleys ( PHOTOS ) '' . Huffington Post . Retrieved 2016 - 10 - 17 . Jump up ^
`` History - Struthers , Ohio '' . Strutherscommunity.weebly.com . Retrieved 2016 - 10 - 17 . Jump up ^ `` Youngstown News , Bowladrome Lanes owners reflect on
40 years in the business '' . Vindy.com . Retrieved 2016 - 10 - 17 . Jump up ^ Sterling , Joe ( July 5 , 1077 ) . `` '' Deer Hunter '' Filming Under Way `` .
Weirton Daily Times . West Virginia Division of Culture and History . Jump up ^ `` The Deer Hunter film locations '' . The Worldwide Guide To Movie Locations .
September 7 , 2015 . ^ Jump up to : McLellan , Dennis ( November 16 , 2007 ) . `` Peter Zinner , 88 ; film editor won Oscar for ' The Deer Hunter ' '' . Los
Angeles Times .
</item 3>

<item 4>
The Deer Hunter
His direction to his actors : `` You put the gun to your head , Chris , you shoot , you fall over and Bobby cradles your head . '' Filming locations ( edit )
Lemko Hall St. Theodosius Russian Orthodox Cathedral , in the Tremont neighborhood of Cleveland , Ohio . The name plaque is clearly visible in one scene . Lemko
Hall , Cleveland , Ohio . Also located in Tremont , the wedding banquet was filmed here . The name is clearly visible in one scene . U.S. Steel Central Furnaces
in Cleveland , Ohio . Opening sequence steel mill scenes . Patpong , Bangkok , Thailand , the area used to represent Saigon 's red light district . Sai Yok ,
Kanchanaburi Province , Thailand Mount Baker - Snoqualmie National Forest and Nooksack Falls in the North Cascades range of , Washington , deer hunting scenes .
Steubenville , Ohio , for some mill and neighborhood shots . Struthers , Ohio , for external house and long - range road shots .
</item 4>

<item 5>
The Deer Hunter
Struthers , Ohio , for external house and long - range road shots . Also including , the town 's bowling alley is the Bowladrome Lanes , located at 56 State
Street , Struthers , Ohio . Weirton , West Virginia , for mill and trailer shots . River Kwai , Thailand , Prison camp and initial Russian roulette scene .
Post-production ( edit ) By this point , The Deer Hunter had cost $13 million and the film still had to go through an arduous post-production . Film editor
Peter Zinner was given 600,000 feet of printed film to edit , a monumental task at the time . Producers Spikings and Deeley were pleased with the first cut ,
which ran for three and a half hours . `` We were thrilled by what we saw , '' wrote Deeley , `` and knew that within the three and a half hours we watched
there was a riveting film . '' Executives from Universal , including Lew Wasserman and Sid Sheinberg , were not very enthusiastic . `` I think they were shocked
, '' recalled Spikings . `` What really upset them was ' God Bless America ' .
</item 5>

<item 6>
The Deer Hunter
The cast and crew viewed large amounts of news footage from the war to ensure authenticity . The film was shot over a period of six months . The Clairton scenes
comprise footage shot in eight different towns in four states : West Virginia , Pennsylvania , Washington , and Ohio . The initial budget of the film was $8.5
million . Meryl Streep accepted the role of the `` vague , stock girlfriend '' , in order to remain for the duration of filming with John Cazale , who had been
diagnosed with lung cancer . De Niro had spotted Streep in her stage production of The Cherry Orchard and had suggested that she play his girlfriend Linda .
Before the beginning of principal photography , Deeley had a meeting with the film 's appointed line producer Robert Relyea . Deeley hired Relyea after meeting
him on the set of Bullitt ( 1968 ) and was impressed with his experience . However , Relyea declined the job , refusing to disclose his reason why . Deeley
suspected that Relyea sensed in director Cimino something that would have made production difficult .
</item 6>

<item 7>
The Deer Hunter
2 Washburn 's claim 3.2. 3 Deeley 's reaction to the revised script 4 Filming 4.1 The wedding scenes 4.2 The bar and the steel mill 4.3 Hunting the deer 4.4
Vietnam and the Russian roulette scenes 4.5 Filming locations 5 Post-production 5.1 Sound design 5.2 Previews 6 Soundtrack 6.1 Selected tracks 7 Release 8
Analysis 8.1 Controversy over Russian roulette 8.1. 1 Critics ' response 8.1.
</item 7>

<item 8>
The Deer Hunter
In the end , it took 51 minutes . Deeley believes that Cimino always planned to make this prologue last for an hour , and `` the plan was to be advanced by
stealth rather than straight dealing . '' At this point in the production , nearly halfway through principal photography , Cimino was already over budget , and
producer Spikings could tell from the script that shooting the extended scene could sink the project . The bar and the steel mill ( edit ) The bar was specially
constructed in an empty storefront in Mingo Junction , Ohio for $25,000 ; it later became an actual saloon for local steel mill workers . U.S. Steel allowed
filming inside its Cleveland mill , including placing the actors around the furnace floor , only after securing a $5 million insurance policy . Other filming
took place in Pittsburgh . Hunting the Deer ( edit ) The first deer to be shot was depicted in a `` gruesome close - up '' , although he was hit with a
tranquilizer dart . The stag that Michael allows to get away later was the same one used on TV commercials for the Connecticut Life Insurance Company .
</item 8>

<item 9>
The Deer Hunter
The Deer Hunter - wikipedia The Deer Hunter Jump to : navigation , search For other uses , see Deer Hunter ( disambiguation ) . The Deer Hunter UK theatrical
release poster Directed by Michael Cimino Produced by Barry Spikings Michael Deeley Michael Cimino John Peverall Screenplay by Deric Washburn Story by Deric
Washburn Michael Cimino Louis Garfinkle Quinn K. Redeker Starring Robert De Niro John Cazale John Savage Meryl Streep Christopher Walken Music by Stanley Myers
Cinematography Vilmos Zsigmond Edited by Peter Zinner Production company EMI Distributed by Universal Pictures Release date December 8 , 1978 ( 1978 - 12 - 08 )
( Los Angeles ) February 23 , 1979 ( 1979 - 02 - 23 ) ( United States ) Running time 183 minutes Country United States Language English Russian Vietnamese
French Budget $15 million Box office $49 million The Deer Hunter is a 1978 American epic war drama film co-written and directed by Michael Cimino about a trio
of Russian American steelworkers whose lives are changed forever after they fight in the Vietnam War .
</item 9>

<item 10>
The Deer Hunter
1 Critics ' response 8.1. 2 Cast and crew response 8.2 Director Cimino 's autobiographical intent 8.3 Homosocial bonding 8.4 Coda of `` God Bless America '' 9
Reception 9.1 Top - ten lists 9.2 Revisionism following Heaven 's Gate 10 Awards 10.1 Lead - up to awards season 10.2 51st Academy Awards 10.3 Golden Globes
10.4 Complete list of awards 11 Legacy 11.1 Honors and recognition 12 Home media 13 See also 14 References 14.1 Annotations 14.2 Notes 14.3 Bibliography 14.4
Further reading 15 External links Plot ( edit ) In the small working class town of Clairton , Pennsylvania , in late 1967 , steel workers Mike Vronsky , Steven
Pushkov , and Nick Chevotarevich , with the support of their friends and co-workers Stan and Peter `` Axel '' Axelrod and local bar owner and friend John Welsh
, prepare for two rites of passage : marriage and military service .
</item 10>
=== END ITEM id=385905ae2cfbe32f ===
