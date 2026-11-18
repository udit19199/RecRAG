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

=== ITEM id=85c617f6883fa512 ===
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
Alicia and Chris get drunk in and vandalize the abandoned home of a wealthy family . Strand recruits Nick for an escape plot . Travis convinces Moyers ' squad
to take him to the hospital to check on his friends . While en route , Moyers encourages Travis to shoot a zombie , but Travis is emotionally unable to pull the
trigger . The soldiers stop to assist another squad in a building infested by zombies , and most of those soldiers , including Moyers , are overcome . The few
survivors flee and drop off Travis near the Safe Zone . Travis learns that Daniel tortured Adams into revealing what `` Cobalt '' means : in the morning , all
civilians will be killed , and the guardsmen will evacuate the city . Griselda dies of septic shock at the hospital ; Liza shoots her brain to prevent
reanimation . Daniel visits a nearby sports arena to verify Adams ' story that it was sealed with 2,000 now - zombified civilians inside .
</item 1>

<item 2>
Fear the Walking Dead (season 1)
Chris ' bus is trapped in a traffic jam caused by a zombie shooting by the police . He films the event and joins in a protest against that and other recent
fatal police shootings , when Travis and Liza meet up with him . A riot erupts after police shoot down another zombie , but the three Manawas find refuge with
the Salazars in their gated barbershop - home . Travis tells Madison to take the kids to the desert without him ; he will catch up . Alicia witnesses her
zombified neighbor , Mr. Dawson , attacking Mrs. Cruz across the street , but Madison prevents her from intervening . The group inside the barbershop remains
trapped , while the riot outside intensifies . `` The Dog '' Adam Davidson Jack LoGiudice September 13 , 2015 ( 2015 - 09 - 13 ) 7.19 While a riot rages outside
, a mob sets fire to the store adjoining the barbershop , forcing the Salazars and Manawas to flee . The group reaches Travis ' truck and escapes , but not
before Griselda is injured by a collapsing scaffold .
</item 2>

<item 3>
Graph query row:
entity: Not Fade Away
relationship: PART_OF
details: {}
related: Fear the Walking Dead (season 1)
</item 3>

<item 4>
Fear the Walking Dead (season 1)
After Travis and Madison arrive , the zombified Calvin attacks them . Nick runs over Calvin repeatedly with Travis ' truck , and the three watch in disbelief as
the mutilated Calvin is still able to turn his head towards them . `` So Close , Yet So Far '' Adam Davidson Marco Ramirez August 30 , 2015 ( 2015 - 08 - 30 )
8.18 After the run - in with zombified Calvin , Nick , Madison , and Travis choose to flee to the desert , and they want Alicia , Liza , and Chris to follow .
Alicia finds Matt ill in his disheveled house . Travis arrives and sees a bite on Matt , who convinces Alicia to leave . The group returns to Madison 's home to
gather supplies . Nick suffers from heroin withdrawal , so Madison drives to her school to get him Oxycodone . There , she encounters Tobias scavenging food . A
zombified Artie tries to bite Tobias , so Madison kills Artie and drives Tobias home . Chris ' bus is trapped in a traffic jam caused by a zombie shooting by
the police .
</item 4>

<item 5>
Fear the Walking Dead (season 1)
1 Talking Dead Fear the Walking Dead Episodes season 1 Web series The Walking Dead Torn Apart Cold Storage Fear the Walking Dead Flight 462 Miscellaneous Video
games Season One `` 400 Days '' Two Michonne Three : A New Frontier Other games Survival Instinct Road to Survival Rides The Walking Dead : The Ride Novels Rise
of the Governor The Road to Woodbury The Fall of the Governor Descent Invasion Search and Destroy Return to Woodbury Characters Lists Comic book series
Television series Aaron Alden Amy Jessie Anderson Andrea Axel Tara Chambler Clementine Daryl Dixon Merle Dixon Dwight Rosita Espinosa Lee Everett King Ezekiel
Abraham Ford Gareth The Governor Gregory Beth Greene Hershel Greene Maggie Greene Carl Grimes Lori Grimes Rick Grimes Dale Horvath Jadis Morgan Jones Lilly
Milton Mamet Caesar Martinez Deanna Monroe Paul `` Jesus '' Monroe Spencer Monroe Morales Michonne Negan Olivia The Prisoners Carol Peletier Sophia Peletier
Eugene Porter Glenn Rhee Lizzie and Mika Samuels Simon Gabriel Stokes Bob Stookey T - Dog Shane Walsh Sasha Williams Tyreese Williams Retrieved from ``
https://en.wikipedia.org/w/index.
</item 5>

<item 6>
Fear the Walking Dead (season 1)
`` Not Fade Away '' Kari Skogland Meaghan Oppenheimer September 20 , 2015 ( 2015 - 09 - 20 ) 6.62 Days after the National Guard quarantines the neighborhood
into a Safe Zone , residents try to live normally . Tensions build under the military rule . Madison is stressed by extra work caused by her home 's being
overcrowded and Travis ' new role as civilian liaison . Chris shows a video to Travis and Madison of a light signaling from the Dead Zone . Travis talks Doug
into getting mental help . Liza medically helps neighbors . Nick steals morphine from Hector via IV drip . Ofelia kisses Adams , who was unable to get Griselda
's medicine . Madison exchanges signals with the flashing light in the Dead Zone . She sneaks outside the fence to investigate and finds evidence that the
guardsmen killed civilians , even the uninfected . Travis learns that Doug has been hospitalized for his mental issues . Dr. Exner determines that Liza is not
technically a nurse .
</item 6>

<item 7>
Fear the Walking Dead (season 1)
6 6 `` The Good Man '' Stefan Schwartz Robert Kirkman & Dave Erickson October 4 , 2015 ( 2015 - 10 - 04 ) 6.86 The group drives to the National Guard 's
headquarters to rescue Liza , Griselda , and Nick . Adams agrees to be their guide when let go by Travis . The group infiltrates the base after Daniel distracts
the guards by leading a horde of walkers from the arena . Travis , Madison , Daniel , and Ofelia go inside , while Alicia and Chris stay behind . Meanwhile ,
the walkers breach the perimeter defenses and swarm the base . Travis ' group reach the holding cells and set the detainees free before reuniting with Nick ,
Liza , and Strand . They try to escape through the medical ward , where they discover Dr. Exner has euthanized all of the patients . Dr. Exner tells them of an
escape route before presumably committing suicide . Before they can escape , the group encounters Adams , who shoots Ofelia in the arm . Enraged , Travis
brutally beats Adams and leaves him for dead .
</item 7>

<item 8>
Fear the Walking Dead (season 1)
overall No. in season Title Directed by Written by Original air date U.S. viewers ( millions ) `` Pilot '' Adam Davidson Robert Kirkman & Dave Erickson August
23 , 2015 ( 2015 - 08 - 23 ) 10.13 Nick awakens in a heroin den in an abandoned church , to find his girlfriend Gloria eating a corpse . While fleeing , he is
hit by a car and hospitalized . The doctor tells Madison and Travis that Nick 's claims about the incident are heroin hallucinations , but Travis believes Nick
after visiting the church himself . Alicia becomes more worried about Nick 's chemical dependency . The next day , school closes early due to the high levels of
absenteeism and rumors of an epidemic . Nick escapes from the hospital and meets with Calvin , hoping to learn if the drugs Calvin sold him caused him to
hallucinate in the church . Calvin tries to kill Nick to prevent him from exposing Calvin as a drug dealer or impugning the quality of his drugs . In the
ensuing struggle , Calvin is mortally shot . After Travis and Madison arrive , the zombified Calvin attacks them .
</item 8>

<item 9>
Fear the Walking Dead (season 1)
Unable to reach a hospital , the group drives to Madison 's house , where Nick , Madison , and Alicia temporarily flee when the zombified Mr. Dawson attempts to
enter , attracted by the barking dog Nick had let in . Nick leads Madison and Alicia to the Trans ' house next door , where they take a shotgun . Travis arrives
and is attacked by Mr. Dawson , who is shot and killed by Daniel . All three families decide to stay the night and evacuate in the morning . Nurse Liza tends to
Griselda 's injured foot but notes that Griselda will die if not treated by a doctor . Ofelia tells Daniel they should flee with Travis , but Daniel insists his
family can survive alone and will join his cousin later . The next morning , as the Clarks and Manawas start driving away , the National Guard arrives and
quarantines the block . While Travis says , `` It 's going to get better , '' Daniel laments that it 's , `` too late , '' as he watches a guardsman mark the
neighboring house .
</item 9>

<item 10>
Fear the Walking Dead (season 1)
Dr. Exner determines that Liza is not technically a nurse . Daniel warns Madison of his El Salvador experience , when the sick were taken under the guise of
receiving hospitalization but instead killed . Soldiers take Griselda and Nick to a hospital , but Nick 's family protests his departure . Liza agrees to go to
assist the medical team , despite not wanting to leave her son . Travis retreats to the roof and sees the signal from the Dead Zone . Seconds later , he sees
and hears gunfire , followed by darkness . 5 5 `` Cobalt '' Kari Skogland David Wiener September 27 , 2015 ( 2015 - 09 - 27 ) 6.66 In a military cell , Strand
bribes a guardsman to save feverish Nick from being moved . Liza helps Dr. Exner with patients at the hospital . Chris is devastated that Liza left voluntarily
to help at the hospital , but Travis promises to bring her back . Madison discovers Daniel detaining Adams in the Trans ' basement . Alicia and Chris get drunk
in and vandalize the abandoned home of a wealthy family .
</item 10>
=== END ITEM id=85c617f6883fa512 ===

=== ITEM id=475a10b5738a40b8 ===
Gold sentences and facts:
Sentence 0: The centerpiece of contemporary Thanksgiving in the United States and Canada is a large meal , generally centered on a large roasted turkey . It is served with a variety of side dishes which vary from traditional dishes such as mashed potatoes , stuffing , and cranberry sauce , to ones that reflect regional or cultural heritage . The majority of the dishes in the traditional American version of Thanksgiving dinner are made from foods native to the New World , as according to tradition the Pilgrims received these foods , or learned how to grow them , from the Native Americans . Thanksgiving dinner is the largest eating event in the United States ; people eat more on Thanksgiving than on any other day of the year .
  Fact 0.0: The centerpiece of contemporary Thanksgiving in the United States and Canada is a large meal.
  Fact 0.1: The large meal is generally centered on a large roasted turkey.
  Fact 0.2: The large meal is served with a variety of side dishes.
  Fact 0.3: The side dishes vary from traditional dishes such as mashed potatoes, stuffing, and cranberry sauce.
  Fact 0.4: The side dishes vary to ones that reflect regional or cultural heritage.
  Fact 0.5: The majority of the dishes in the traditional American version of Thanksgiving dinner are made from foods native to the New World.
  Fact 0.6: According to tradition, the Pilgrims received these foods, or learned how to grow them, from the Native Americans.
  Fact 0.7: Thanksgiving dinner is the largest eating event in the United States.
  Fact 0.8: People eat more on Thanksgiving than on any other day of the year.

Retrieved items:
<item 1>
Thanksgiving dinner
There are also regional differences as to the stuffing or dressing traditionally served with the turkey . The traditional version has bread cubes , sage , onion
and celery . Southerners generally make their dressing from cornbread , while those in other parts of the country make stuffing from white , wheat or rye bread
as the base . One or several of the following may be added to the dressing / stuffing : oysters , apples , chestnuts , raisins , and sausages or the turkey 's
giblets . Other dishes reflect the region or cultural background of those who have come together for the meal . For example , Sauerkraut ( among those in the
Mid-Atlantic ; especially Baltimore ) is sometimes served . Many African Americans and Southerners serve baked macaroni and cheese and collard greens , along
with chitterlings and sweet potato pie , while some Italian - Americans often have lasagne on the table and Ashkenazi Jews may serve noodle kugel , a sweet
dessert pudding . Other Jewish families may consume foods commonly associated with Hanukkah , such as latkes or a sufganiyah ; the two holidays are usually in
close proximity and on extremely rare occasions overlap .
</item 1>

<item 2>
Thanksgiving dinner
Copious leftovers are also common following the meal properly . Traditional Thanksgiving foods are sometimes specific to the day of riced potatoes , and
although some of the foods might be seen at any semi-formal meal in the United States , the meal often has something of a ritual or traditional quality . Many
Americans would say it is `` incomplete '' without cranberry sauce , stuffing or dressing , mashed potatoes and gravy , and brussels sprouts . Other commonly
served dishes include winter squash and sweet potatoes , the latter often prepared with sweeteners such as brown sugar , molasses , or marshmallows . Fresh ,
canned , or frozen corn is popular and green beans are frequently served ; in particular , green bean casserole , a product invented in 1955 by the Campbell
Soup Company to promote use of its cream of mushroom soup , has become a Thanksgiving standard . A fresh salad may be included , especially on the West Coast .
Bread rolls or biscuits and cornbread , especially in the South and parts of New England , are served . For dessert , various pies are usually served ,
particularly pumpkin pie , though apple pie , mincemeat pie , sweet potato pie , and pecan pie are often served as well .
</item 2>

<item 3>
Thanksgiving dinner
It is not unheard of for Mexican Americans to serve their turkey with mole and roasted corn . In Puerto Rico , the Thanksgiving meal is completed with arroz con
gandules ( rice with pigeon peas ) or arroz con maiz ( rice with corn ) , pasteles ( root tamales ) stuffed with turkey , pumpkin - coconut crème caramel , corn
bread with longaniza , potato salad , roasted white sweet potatoes and Spanish sparkling hard cider . Turkey in Puerto Rico is stuffed with mofongo . Cuban -
Americans traditionally serve the turkey alongside a small roasted pork and include white rice and black beans or kidney beans . Vegetarians or vegans have been
known to serve alternative entree centerpieces such as a large vegetable pie or a stuffed and baked pumpkin or tofu substitutes . Many Midwesterners ( such as
Minnesotans ) of Norwegian or Scandinavian descent set the table with lefse . Beverages ( edit ) The beverages at Thanksgiving can vary as much as the side
dishes , often depending on who is present at the table and their tastes . Spirits or cocktails sometimes may be served before the main meal .
</item 3>

<item 4>
Thanksgiving dinner
Thanksgiving dinner - wikipedia Thanksgiving dinner Jump to : navigation , search ( Almost Traditional ) Thanksgiving Dinner The centerpiece of contemporary
Thanksgiving in the United States and Canada is a large meal , generally centered on a large roasted turkey . It is served with a variety of side dishes which
vary from traditional dishes such as mashed potatoes , stuffing , and cranberry sauce , to ones that reflect regional or cultural heritage . The majority of the
dishes in the traditional American version of Thanksgiving dinner are made from foods native to the New World , as according to tradition the Pilgrims received
these foods , or learned how to grow them , from the Native Americans . Thanksgiving dinner is the largest eating event in the United States ; people eat more
on Thanksgiving than on any other day of the year .
</item 4>

<item 5>
Thanksgiving dinner
Alexander Hamilton proclaimed that no `` Citizen of the United States should refrain from turkey on Thanksgiving Day , '' and Benjamin Franklin had high regard
for the wild turkey as an American icon , but turkey was uncommon as Thanksgiving fare until after 1800 . By 1857 , turkey had become part of the traditional
dinner in New England . A Thanksgiving Day dinner served to the Civilian Conservation Corps in 1935 included : pickles , green olives , celery , roast turkey ,
oyster stew , cranberry sauce , giblet gravy , dressing , creamed asparagus tips , snowflake potatoes , baked carrots , hot rolls , fruit salad , mince meat pie
, fruit cake , candies , grapes , apples , clams , fish , and many other food harvests . French drip coffee , cigars and cigarettes . The White House Cook Book
, 1887 , by Mrs. F.L. Gillette , et al.
</item 5>

<item 6>
Thanksgiving dinner
( Pumpkin pie was forbidden because it contained eggs . ) The National Poultry and Egg Board furiously lobbied the President to cease promoting the plan ; it
culminated in a truce at the National Thanksgiving Turkey Presentation shortly before Thanksgiving . Turkey was no longer forbidden , but Eggless Thursdays
remained for the rest of the year , meaning no pumpkin pie was served at the White House dinner that year . Turkey ( edit ) Oven roasted turkey Turkey is the
most common main dish of a Thanksgiving dinner , to the point where Thanksgiving is sometimes colloquially called `` Turkey Day . '' In 2006 , American turkey
growers were expected to raise 270 million turkeys , to be processed into five billion pounds of turkey meat valued at almost $8 billion , with one third of all
turkey consumption occurring in the Thanksgiving - Christmas season , and a per capita consumption of almost 18 pounds ( 8.2 kg ) . The Broad Breasted White
turkey is particularly bred for Thanksgiving dinner and similar large feasts ; its large size ( specimens can grow to over 40 pounds ) and meat content make it
ideal for such situations , although the breed must be artificially bred and suffers from health problems due to its size .
</item 6>

<item 7>
Thanksgiving dinner
In Alaskan villages , whale meat is sometimes eaten. Irish immigrants have been known to have prime rib of beef as their centerpiece since beef in Ireland was
once a rarity ; families would save up money for this dish to signify newfound prosperity and hope . In the United States , a globalist approach to Thanksgiving
has become common with the impact of immigration . Basic `` Thanksgiving '' ingredients , or the intent of the holiday , can be transformed to a variety of
dishes by using flavors , techniques , and traditions from their own cuisines . Others celebrate the holiday with a variety of dishes particularly when there is
a crowd to be fed , guest 's tastes vary and considering the financial means available . Side dishes ( edit ) A Thanksgiving meal in New England Green bean
casserole Many offerings are typically served alongside the main dish -- so many that , because of the amount of food , the Thanksgiving meal is sometimes
served midday or early afternoon to make time for all the eating , and preparation may begin at dawn or on days prior . Copious leftovers are also common
following the meal properly .
</item 7>

<item 8>
Thanksgiving dinner
Contents ( hide ) 1 Historical menus 2 Turkey 3 Alternatives to turkey 4 Side dishes 5 Beverages 6 See also 7 References Historical menus ( edit ) 1943
Thanksgiving Day dinner menu from USS Wake Island ( CVE - 65 ) According to what traditionally is known as `` The First Thanksgiving , '' the 1621 feast between
the Pilgrims and the Wampanoag at Plymouth Colony contained waterfowl , venison , ham , lobster , clams , berries , fruit , pumpkin , and squash . William
Bradford noted that , `` besides waterfowl , there was great store of wild turkeys , of which they took many . '' Many of the foods that were included in the
first feast ( except , notably , the seafood ) have since gone on to become staples of the modern Thanksgiving dinner . Early feasts of the Order of Good Cheer
, a French Canadian predecessor to the modern Thanksgiving , featured a potluck dinner with freshly - hunted fowl , game , and fish , hunted and shared by both
French Canadians and local natives . The use of the turkey in the US for Thanksgiving precedes Lincoln 's nationalization of the holiday in 1863 .
</item 8>

<item 9>
Thanksgiving dinner
Ham is often served alongside turkey in many non-traditional households . Goose and duck , foods which were traditional European centerpieces of Christmas
dinners before being displaced , are now sometimes served in place of the Thanksgiving turkey . Sometimes , fowl native to the region where the meal is taking
place is used ; for example , an article in Texas Monthly magazine suggested quail as the main dish for a Texan Thanksgiving feast . John Madden , who appeared
on television for the NFL Thanksgiving Day game from 1981 to 2001 , frequently advocated his fondness for the turducken , deboned turkey , duck and chicken
nested inside each other then cooked . In a few areas of the West Coast of the United States , Dungeness crab is common as an alternate main dish , as crab
season starts in early November . Similarly , Thanksgiving falls within deer hunting season in the Northeastern United States , which encourages the use of
venison as a centerpiece . Vegetarians or vegans may have a tofu , wheat gluten or lentil - based substitute ; or stuffed squash . In Alaskan villages , whale
meat is sometimes eaten.
</item 9>

<item 10>
Thanksgiving dinner
Most Thanksgiving turkeys are stuffed with a bread - based mixture and roasted . Sage is the traditional herb added to the stuffing , along with chopped celery
, carrots , and onions . Other ingredients , such as chopped chestnuts or other tree nuts , crumbled sausage or bacon , cranberries , raisins , or apples , may
be added to stuffing . If this mixture is prepared outside the bird , it may be known as dressing . Deep - fried turkey is rising in popularity due to its
shorter preparation time , but carries safety risks . The consumption of turkey on Thanksgiving is so ingrained in American culture that each year since 1947 ,
the National Turkey Federation has presented a live turkey to the President of the United States prior to each Thanksgiving . These turkeys were initially
slaughtered and eaten for the President 's Thanksgiving dinner ; since 1989 , the presented turkeys have typically been given a mock pardon to great fanfare and
sent to a park to live out the rest of their usually short natural lives . Alternatives to turkey ( edit ) Non-traditional foods other than turkey are sometimes
served as the main dish for a Thanksgiving dinner . Ham is often served alongside turkey in many non-traditional households .
</item 10>
=== END ITEM id=475a10b5738a40b8 ===

=== ITEM id=c8e2e2a5b0dca441 ===
Gold sentences and facts:
Sentence 0: Much of the humus in most soils has persisted for more than 100 years , rather than having been decomposed into CO , and can be regarded as stable ; this organic matter has been protected from decomposition by microbial or enzyme action because it is hidden ( occluded ) inside small aggregates of soil particles , or tightly sorbed or complexed ) to clays . Most humus that is not protected in this way is decomposed within 10 years and can be regarded as less stable or more labile . Stable humus contributes few plant - available nutrients in soil , but it helps maintain its physical structure . A very stable form of humus is formed from the slow oxidation of soil carbon after the incorporation of finely powdered charcoal into the topsoil . This process is speculated to have been important in the formation of the very fertile Amazonian terra preta do Indio .
  Fact 0.0: Much of the humus in most soils has persisted for more than 100 years rather than having been decomposed into CO.
  Fact 0.1: The long-persisting humus in most soils can be regarded as stable.
  Fact 0.2: Much of the humus in most soils has been protected from decomposition by microbial or enzyme action.
  Fact 0.3: Protected humus is hidden (occluded) inside small aggregates of soil particles, or is tightly sorbed or complexed to clays.
  Fact 0.4: Most humus that is not protected in this way is decomposed within 10 years.
  Fact 0.5: Humus that is not protected in this way can be regarded as less stable or more labile.
  Fact 0.6: Stable humus contributes few plant-available nutrients in soil.
  Fact 0.7: Stable humus helps maintain the physical structure of soil.
  Fact 0.8: A very stable form of humus is formed from the slow oxidation of soil carbon after finely powdered charcoal is incorporated into the topsoil.
  Fact 0.9: The formation of the very stable humus from charcoal is speculated to have been important in the formation of the very fertile Amazonian terra preta do Indio.

Retrieved items:
<item 1>
Humus
Most soils have three major horizons : the surface horizon ( A ) , the subsoil ( B ) , and the substratum ( C ) . Some soils have an organic horizon ( O ) on
the surface , but this horizon can also be buried . The master horizon ( E ) is used for subsurface horizons that have significantly lost minerals ( eluviation
) . Bedrock , which is not soil , uses the letter R . Benefits of soil organic matter and humus ( edit ) The importance of chemically stable humus is thought by
some to be the fertility it provides to soils in both a physical and chemical sense , though some agricultural experts put a greater focus on other features of
it , such as its ability to suppress disease . It helps the soil retain moisture by increasing microporosity , and encourages the formation of good soil
structure . The incorporation of oxygen into large organic molecular assemblages generates many active , negatively charged sites that bind to positively
charged ions ( cations ) of plant nutrients , making them more available to the plant by way of ion exchange . Humus allows soil organisms to feed and reproduce
, and is often described as the `` life - force '' of the soil .
</item 1>

<item 2>
Humus
Humus has many nutrients that improve the health of soil , nitrogen being the most important . The ratio of carbon to nitrogen ( C : N ) of humus is 10 : 1 .
Contents ( hide ) 1 Description 2 Humification 3 Stability 4 Horizons 5 Benefits of soil organic matter and humus 6 See also 7 References 8 External links
Description ( edit ) It is difficult to define humus precisely because it is a very complex substance which is not fully understood . Humus is different from
decomposing soil organic matter . The latter looks rough and has visible remains of the original plant or animal matter . Fully humified humus , on the contrary
, has a uniformly dark , spongy , and jelly - like appearance , and is amorphous ; it may gradually decompose over several years or persist for millennia . It
has no determinate shape , structure , or quality . However , when examined under a microscope , humus may reveal tiny plant , animal , or microbial remains
that have been mechanically , but not chemically , degraded . This suggests an ambiguous boundary between humus and soil organic matter .
</item 2>

<item 3>
Humus
The process that converts soil organic matter into humus feeds the population of microorganisms and other creatures in the soil , and thus maintains high and
healthy levels of soil life . The rate at which soil organic matter is converted into humus promotes ( when fast ) or limits ( when slow ) the coexistence of
plants , animals , and microorganisms in the soil . Effective humus and stable humus are additional sources of nutrients for microbes : the former provides a
readily available supply and the latter acts as a longeval storage reservoir . Decomposition of dead plant material causes complex organic compounds to be
slowly oxidized ( lignin - like humus ) or to decompose into simpler forms ( sugars and amino sugars , and aliphatic and phenolic organic acids ) , which are
further transformed into microbial biomass ( microbial humus ) or reorganized , and further oxidized , into humic assemblages ( fulvic and humic acids ) , which
bind to clay minerals and metal hydroxides . The ability of plants to absorb humic substances with their roots and metabolize them has been long debated . There
is now a consensus that humus functions hormonally rather than simply nutritionally in plant physiology .
</item 3>

<item 4>
Humus
Soil Biology and Biochemistry . 26 ( 2 ) : 183 -- 220 . doi : 10.1016 / 0038 - 0717 ( 94 ) 90161 - 9 . Jump up ^ Humintech ® Definition Of Soil Organic Matter &
Humic Acids Based Products Jump up ^ Brady , N.C. ; Weil , R.R. ( 1999 ) . The nature and properties of soils . Upper Saddle River , N.J. : Prentice Hall , Inc
. Jump up ^ Soil biology Jump up ^ Berg , B. , McClaugherty , C. , 2007 . Plant litter : decomposition , humus formation , carbon sequestration , 2nd ed .
Springer , 338 pp. , ISBN 3 - 540 - 74922 - 5 Jump up ^ Levin , L. ; Forchiassin , F. ; Ramos , A.M. ( 2002 ) . `` Copper induction of lignin - modifying
enzymes in the white - rot fungus Trametes trogii '' . Mycologia. 94 : 377 -- 383 .
</item 4>

<item 5>
Humus
Humus is a colloidal substance and increases the cation exchange capacity of soil , hence its ability to store nutrients by chelation . While these nutrient
cations are available to plants , they are held in the soil and prevented from being leached by rain or irrigation . Humus can hold the equivalent of 80 -- 90 %
of its weight in moisture , and therefore increases the soil 's capacity to withstand drought . The biochemical structure of humus enables it to moderate , i.e.
buffer , excessive acidic or alkaline soil conditions . During humification , microbes secrete sticky , gum - like mucilages ; these contribute to the crumby
structure ( tilth ) of the soil by adhering particles together and allowing greater aeration of the soil . Toxic substances such as heavy metals and excess
nutrients can be chelated , i.e. , bound to the organic molecules of humus , and so prevented from leaching away . The dark , usually brown or black , color of
humus helps to warm cold soils in Spring .
</item 5>

<item 6>
Humus
This suggests an ambiguous boundary between humus and soil organic matter . While distinct , humus is an integral part of soil organic matter . Humification (
edit ) Microorganisms decompose a large portion of the soil organic matter into inorganic minerals that the roots of plants can absorb as nutrients . This
process is termed `` mineralization '' . In this process , nitrogen ( nitrogen cycle ) and the other nutrients ( nutrient cycle ) in the decomposed organic
matter are recycled . Depending on the conditions in which the decomposition occurs , a fraction of the organic matter does not mineralize , and instead is
transformed by a process called `` humification '' into concatenations of organic polymers . Because these organic polymers are resistant to the action of
microorganisms , they are stable , and constitute humus . This stability implies that humus integrates into the permanent structure of the soil , thereby
improving it . Humification can occur naturally in soil or artificially in the production of compost . Organic matter is humified by a combination of
saprotrophic fungi , bacteria , microbes and animals such as earthworms , nematodes , protozoa , and arthropods .
</item 6>

<item 7>
Humus
Humus has a characteristic black or dark brown color and is an accumulation of organic carbon . Besides the three major soil horizons of ( A ) surface / topsoil
, ( B ) subsoil , and ( C ) substratum , some soils have an organic horizon ( O ) on the very surface . Hard bedrock ( R ) is not in a strict sense soil . In
soil science , humus ( derived in 1790 -- 1800 from the Latin humus for earth , ground ) denominates the fraction of soil organic matter that is amorphous and
without the `` cellular cake structure characteristic of plants , micro-organisms or animals . '' Humus significantly affects the bulk density of soil and
contributes to its retention of moisture and nutrients . In agriculture , `` humus '' sometimes also is used to describe mature or natural compost extracted
from a woodland or other spontaneous source for use as a soil conditioner . It is also used to describe a topsoil horizon that contains organic matter ( humus
type , humus form , humus profile ) . Humus is the dark organic matter that forms in soil when dead plant and animal matter decays .
</item 7>

<item 8>
Humus
, 2011 . Accessed 24 Nov 2011 . < http://www.britannica.com/EBchecked/topic/276408/humus > . Jump up ^ Chertov , O.G. ; Kornarov , A.S. ; Crocker , G. ; Grace ,
P. ; Klir , J. ; Körschens , M. ; Poulton , P.R. ; Richter , D. ( 1997 ) . `` Simulating trends of soil organic carbon in seven long - term experiments using
the SOMM model of the humus types '' . Geoderma. 81 : 121 -- 135 . doi : 10.1016 / S0016 - 7061 ( 97 ) 00085 - 2 . Jump up ^ Baritz , R. , 2003 . Humus forms in
forests of the northern German lowlands . Schweizerbart , Stuttgart , Germany , 145 pp. ( 2 ) Jump up ^ Bunting , B.T. ; Lundberg , J. ( 1995 ) .
</item 8>

<item 9>
Humus is the dark, amorphous fraction of soil organic matter formed as microbes and soil animals transform dead plant and animal material. It can persist for
decades to millennia, especially when protected within soil aggregates or bound to clay. Humus improves soil structure and aeration, holds water and nutrients,
increases cation-exchange capacity, buffers acidity, and supports soil life. It is associated with the organic-rich surface horizon and is sometimes used to
mean mature compost.
</item 9>

<item 10>
Humus
Stability ( edit ) Much of the humus in most soils has persisted for more than 100 years , rather than having been decomposed into CO , and can be regarded as
stable ; this organic matter has been protected from decomposition by microbial or enzyme action because it is hidden ( occluded ) inside small aggregates of
soil particles , or tightly sorbed or complexed ) to clays . Most humus that is not protected in this way is decomposed within 10 years and can be regarded as
less stable or more labile . Stable humus contributes few plant - available nutrients in soil , but it helps maintain its physical structure . A very stable
form of humus is formed from the slow oxidation of soil carbon after the incorporation of finely powdered charcoal into the topsoil . This process is speculated
to have been important in the formation of the very fertile Amazonian terra preta do Indio . Horizons ( edit ) Humus has a characteristic black or dark brown
color and is organic due to an accumulation of organic carbon . Soil scientists use the capital letters O , A , B , C , and E to identify the master horizons ,
and lowercase letters for distinctions of these horizons .
</item 10>
=== END ITEM id=c8e2e2a5b0dca441 ===

=== ITEM id=a0f8f8a6d8eab66b ===
Gold sentences and facts:
Sentence 0: The Pan-American Highway is a network of roads measuring about 30,000 kilometres ( 19,000 mi ) in total length . Except for a rainforest break of approximately 160 km ( 100 mi ) , called the Darién Gap , the road links almost all of the Pacific coastal countries of the Americas in a connected highway system . According to Guinness World Records , the Pan-American Highway is the world 's longest `` motorable road '' . However , because of the Darién Gap , it is not possible to cross between South America and Central America with conventional highway vehicles . Without an all - terrain vehicle , it is necessary to circumnavigate this terrestrial stretch by sea .
  Fact 0.0: The Pan-American Highway is a network of roads.
  Fact 0.1: The Pan-American Highway measures about 30,000 kilometres (19,000 mi) in total length.
  Fact 0.2: The Pan-American Highway has a rainforest break of approximately 160 km (100 mi).
  Fact 0.3: The rainforest break of the Pan-American Highway is called the Darién Gap.
  Fact 0.4: Apart from the Darién Gap, the Pan-American Highway links almost all of the Pacific coastal countries of the Americas.
  Fact 0.5: The Pan-American Highway links those countries in a connected highway system.
  Fact 0.6: According to Guinness World Records, the Pan-American Highway is the world's longest 'motorable road'.
  Fact 0.7: Because of the Darién Gap, it is not possible to cross between South America and Central America with conventional highway vehicles.
  Fact 0.8: Without an all-terrain vehicle, it is necessary to go around the terrestrial stretch of the Darién Gap by sea.

Retrieved items:
<item 1>
Pan-American Highway
The Independent ( UK ) . Retrieved 3 August 2010 . Sources ( edit ) Plan Federal Highway System , New York Times May 15 , 1932 page XX7 Reported from the Motor
World , New York Times January 26 , 1936 page XX6 Hemisphere Road is Nearer Reality , New York Times January 7 , 1953 page 58 1997 -- 98 AAA Caribbean , Central
America and South America map Popular Mechanics , March 1943 , Longest Road In The World External links ( edit ) Media related to Pan-American Highway at
Wikimedia Commons Pan-American Highway travel guide from Wikivoyage Geographic data related to Pan-American Highway at OpenStreetMap Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Pan-American_Highway&oldid=838034809 '' Categories : Pan-American Highway Hidden categories : Articles containing
French - language text Articles containing Portuguese - language text Articles containing Spanish - language text Webarchive template wayback links CS1 Spanish
- language sources ( es ) All articles with dead external links Articles with dead external links from March 2018 Articles with permanently dead external links
Pages using div col with deprecated parameters All articles with unsourced statements Articles with
</item 1>

<item 2>
Pan-American Highway
The road had formerly ended at Cañita , Panama , 110 miles ( 180 km ) north of its current end . United States government funding was particularly significant
to complete the high - level Bridge of the Americas over the Panama Canal , during the years when the canal was administered by the United States . Belize was
supposedly included in the route at one time , after it switched to driving on the right . Prior to independence , as British Honduras , it was the only Central
American country to drive on the left side of the road . Darién Gap ( edit ) Map of the Darién Gap and the break in the Pan-American Highway between Yaviza ,
Panama and Turbo , Colombia Main article : Darién Gap The Pan-American Highway is interrupted between Panama and Colombia by a 100 km ( 60 mi ) stretch of
marshland known as the Darién Gap . The highway terminates at Turbo , Colombia and Yaviza , Panama . Because of swamps , marshes , and rivers , construction
would be very expensive . Efforts have been made for decades to eliminate the gap in the Pan-American highway , but have been controversial .
</item 2>

<item 3>
Pan-American Highway
Travelers would then be able to ferry from Key West to Havana , drive to the eastern tip of Cuba , ferry to Haiti , drive through Haiti and the Dominican
Republic , and ferry again to Puerto Rico . Included in this system would also be a ferry from the western tip of Cuba to the Yucatán Peninsula . Mexico has
already surveyed a route which will run across the Yucatán , Campeche , and Chiapas to San Cristobal de Las Casas , on the Pan American Highway . ( `` The Pan
American Highway System '' by Travel Division Pan American Union , Washington D.C. October 1947 ) Art and culture ( edit ) Travel writer Tim Cahill wrote a book
, Road Fever , about his record - setting 24 - day drive from Ushuaia in the Argentine province of Tierra del Fuego to Prudhoe Bay in the U.S. state of Alaska
with professional long - distance driver Garry Sowerby , much of their route following the Pan-American Highway . In the British motoring show Top Gear , the
presenters drove on a section of the road in their off - road vehicles in the Bolivian Special .
</item 3>

<item 4>
Pan-American Highway
Panamericana -- Pan American Highway -- in Pichincha , Ecuador , near Cashapamba Panamericana -- northern Peru near Pacasmayo Panamericana near Puerto De Lomas
, Peru Panamericana in the Atacama Desert northern Chile GMC Suburban traveled all of the Pan-American Highway . Patagonia , Chile . Panamericana - Pan American
Highway - in the Cordillera de los Andes , southern Ecuador , near to Catacocha , 2500m / 8000 ft Elev . National Route 7 -- Pan-American Highway in Junín ,
Buenos Aires , Argentina See also ( edit ) CANAMEX Corridor Continental 1 Trans - Siberian Highway Buenos Aires References ( edit ) Jump up ^ French : Route /
Autoroute Panaméricaine / Transaméricaine , Portuguese : Rodovia / Auto - estrada Pan-americana , Spanish : Autopista / Carretera / Ruta Panamericana Jump up ^
`` A Gap in the Andes : Image of the Day '' . earthobservatory.nasa.gov. 2015 - 04 - 02 .
</item 4>

<item 5>
Pan-American Highway
Pan-American Highway - wikipedia Pan-American Highway Jump to : navigation , search `` Panamericana '' redirects here . For the film , see Panamericana ( film )
. Not to be confused with CANAMEX Corridor . The Pan-American Highway from Prudhoe Bay , U.S.A. to Quellón , Chile and Ushuaia , Argentina , with official and
unofficial routes shown in South America , and a few selected unofficial routes shown through the United States and Canada as they existed in the early 1960 's
. In 1966 the new U.S. Interstate highway system brought official status to most previously unofficial routes in the lower 48 states . The Pan-American Highway
is a network of roads measuring about 30,000 kilometres ( 19,000 mi ) in total length . Except for a rainforest break of approximately 160 km ( 100 mi ) ,
called the Darién Gap , the road links almost all of the Pacific coastal countries of the Americas in a connected highway system . According to Guinness World
Records , the Pan-American Highway is the world 's longest `` motorable road '' .
</item 5>

<item 6>
Pan-American Highway
They barely succeeded in thrashing a passage through the extreme terrain . In 1979 a team led by Mark Smith drove standard production CJ7 - model Jeeps from
South to North , traversing the Gap - with difficulty . In June 1984 , Loren and Patty Upton took 741 days slogging , winching , chopping and digging their way
through the inhospitable jungles of the Darién Gap . One proposed option to bridge the gap is a short ferry link from Colombia to a new ferry port in Panama ,
with an extension of the existing Panama highway that would complete the highway without violating these environmental concerns . Southern section ( edit ) Main
article : Pan-American Highway ( South America ) A Vía PanAm shield sign is sometimes found on routes in South American countries ( such as Chile ) associated
with the Pan-American Highway . Sculpture of a native man standing at the entrance of Fusagasugá , Colombia , over the Highway 40 . Colombia and Venezuela (
edit ) The southern part of the highway begins in northwestern Colombia , from where it follows Colombia Highway 62 to Medellín .
</item 6>

<item 7>
Pan-American Highway
However , because of the Darién Gap , it is not possible to cross between South America and Central America with conventional highway vehicles . Without an all
- terrain vehicle , it is necessary to circumnavigate this terrestrial stretch by sea . The Pan-American Highway passes through many diverse climates and
ecological types , from dense jungles , to arid deserts , some of which are passable only during the dry season , and in many regions driving is occasionally
hazardous .
</item 7>

<item 8>
Pan-American Highway
In 2003 , Kevin Sanders , a long - distance rider , broke the Guinness World Record for the fastest traversal of the highway by motorcycle in 34 days . Photo
gallery ( edit ) The northern end of the Pan-American Highway at Deadhorse , Alaska Interstate 25 ( Pan-American Freeway ) approaching the Big I interchange in
Albuquerque , New Mexico , USA Pan-American Highway through San Martin , El Salvador . Another view of the Pan-American Highway in El Salvador . Pan-American
Highway in El Salvador between Lourdes and Santa Ana ; this flat 1.5 km long straight section can be used as an airstrip and it was used during El Salvador
Civil War . Highway in Guanacaste , Costa Rica ( going towards the Nicaraguan border , still many kilometres away . ) Pan-American Highway in Tres Rios , Costa
Rica , right before the toll plaza ( about 337 more km to go until the Panamanian border ) .
</item 8>

<item 9>
Pan-American Highway
Contents ( hide ) 1 Development and construction 2 Countries served 3 Hemispheric overview 4 Northern section 4.1 Alaska , Canada and the contiguous 48 states
of the United States 4.2 Mexico 4.3 Central America 5 Darién Gap 6 Southern section 6.1 Colombia and Venezuela 6.2 Ecuador , Peru , Chile , Argentina and
Paraguay 6.3 Brazil and Uruguay 6.4 Guyana , Suriname and French Guyana 7 West Indies section 8 Art and culture 9 Photo gallery 10 See also 11 References 12
Sources 13 External links Development and construction ( edit ) The concept of an overland route from one tip of the Americas to the other was originally
proposed at the First Pan-American Conference in 1889 as a railroad ; however , this proposal was never realized . The idea of building a highway emerged at the
Fifth International Conference of American States in 1923 . The first conference regarding construction of the highway occurred on October 5 , 1925 .
</item 9>

<item 10>
Pan-American Highway
The first conference regarding construction of the highway occurred on October 5 , 1925 . Finally , on July 29 , 1937 , Argentina , Bolivia , Chile , Colombia ,
Costa Rica , El Salvador , Guatemala , Honduras , Mexico , Nicaragua , Panama , Peru , Canada , and the United States signed the Convention on the Pan-American
Highway , whereby they agreed to speedy construction , by all adequate means . In 1950 , Mexico became the first Latin American country to complete its portion
of the highway . Countries served ( edit ) Map of the Alaska Highway portion ( in red ) of the Pan-American Highway system . The Northern Pan-American Highway
travels through 9 countries : Canada ( unofficial ) United States ( interstate system official ) Mexico Guatemala El Salvador Honduras Nicaragua Costa Rica
Panama The Southern Pan-American Highway travels through 8 countries : Suriname Guyana Venezuela Colombia Ecuador Peru Chile Argentina Important spurs also lead
into 4 countries : Bolivia Brazil Paraguay Uruguay Hemispheric overview ( edit ) The Pan-American Highway system is physically mostly complete and extends in de
facto terms from Prudhoe Bay , Alaska , in North America to the lower reaches of South America .
</item 10>
=== END ITEM id=a0f8f8a6d8eab66b ===

=== ITEM id=aad4df820b6cc8df ===
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
Graph query row:
entity: Gerald Ford
relationship: RELATED_TO
related: Eagle Scout
</item 2>

<item 3>
Gerald Ford
That year his biological father , whom Ford described as a `` carefree , well - to - do man who did n't really give a damn about the hopes and dreams of his
firstborn son '' , approached Ford while he was waiting tables in a Grand Rapids restaurant . The two `` maintained a sporadic contact '' until Leslie King ,
Sr. 's death in 1941 . Ford said , `` My stepfather was a magnificent person and my mother equally wonderful . So I could n't have written a better prescription
for a superb family upbringing . '' Eagle Scout Gerald Ford ( circled in red ) in 1929 ; Michigan Governor Fred W. Green at far left , holding hat Ford as a
center on the University of Michigan football team , 1933 Ford was involved in the Boy Scouts of America , and earned that program 's highest rank , Eagle Scout
. He is the only Eagle Scout to have ascended to the U.S. Presidency . Ford attended Grand Rapids South High School , where he was a star athlete and captain of
the football team . In 1930 , he was selected to the All - City team of the Grand Rapids City League .
</item 3>

<item 4>
Gerald Ford
An incident in 1975 , when he tripped while exiting Air Force One in Austria , was famously and repeatedly parodied by Chevy Chase , cementing Ford 's image as
a klutz . Pieces of Ford 's common Everyman image have also been attributed to Ford 's inevitable comparison to Nixon , as well as his perceived Midwestern
stodginess and self - deprecation . Civic and fraternal organizations ( edit ) Ford was a member of several civic organizations , including the Junior Chamber
of Commerce ( Jaycees ) , American Legion , AMVETS , Benevolent and Protective Order of Elks , Sons of the Revolution , and Veterans of Foreign Wars .
Freemasonry ( edit ) Ford was initiated into Freemasonry on September 30 , 1949 . He later said in 1975 , `` When I took my obligation as a master mason --
incidentally , with my three younger brothers -- I recalled the value my own father attached to that order . But I had no idea that I would ever be added to the
company of the Father of our Country and 12 other members of the order who also served as Presidents of the United States . ''
</item 4>

<item 5>
Graph query row:
entity: Gerald Ford
relationship: RELATED_TO
related: U.S. Presidency
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
President of the United States
Jump up ^ `` Obama signs protection bill for former presidents '' . The Washington Times . January 10 , 2013 . Retrieved August 14 , 2013 . Jump up ^ `` Shock
and Anger Flash Throughout the United States '' . Associated Press . March 31 , 1981 . Retrieved March 11 , 2011 . Jump up ^ `` Four Presidents '' . Reagan
Presidential Library , National Archives and Records Administration . Retrieved April 3 , 2011 . Jump up ^ `` Biography of Richard M. Nixon '' .
obamawhitehouse.archives.gov . , The White House . Jump up ^ 44 U.S.C. § 2112 Further reading Balogh , Brian and Bruce J. Schulman , eds . Recapturing the Oval
Office : New Historical Approaches to the American Presidency ( Cornell University Press , 2015 ) , 311 pp . Bumiller , Elisabeth ( January 2009 ) . `` Inside
the Presidency '' . National Geographic . 215 ( 1 ) : 130 -- 149 . Couch , Ernie .
</item 7>

<item 8>
Gerald Ford
Bush , Peter Dominick , Robert Griffin , Thomas Kuchel , Mel Laird , Bob Mathias , George Murphy , Dick Poff , Chuck Percy , Al Quie , Charlotte Reid , Hugh
Scott , Bill Steiger , John Tower ( 1968 ) Vacant Title next held by Donald Fraser , Scoop Jackson , Mike Mansfield , John McCormack , Patsy Mink , Ed Muskie ,
Bill Proxmire Preceded by Richard Nixon Republican nominee for President of the United States 1976 Succeeded by Ronald Reagan Political offices Preceded by
Spiro Agnew Vice President of the United States 1973 -- 1974 Succeeded by Nelson Rockefeller Preceded by Richard Nixon President of the United States 1974 --
1977 Succeeded by Jimmy Carter Diplomatic posts Preceded by Valéry Giscard d'Estaing Chair of the Group of Seven 1976 Succeeded by James Callaghan Honorary
titles Preceded by Ronald Reagan Oldest living President of the United States 2004 -- 2006 Succeeded by George H.W.
</item 8>

<item 9>
President of the United States
Boy Scouts of America . Retrieved November 9 , 2017 . Jump up ^ Grier , Peter ( April 25 , 2011 ) . `` The ( not so ) secret history of the White House Easter
Egg Roll '' . The Christian Science Monitor . Archived from the original on July 30 , 2012 . Retrieved July 30 , 2012 . Jump up ^ Hesse , Monica ( November 21 ,
2007 ) . `` Turkey Pardons , The Stuffing of Historic Legend '' . The Washington Post . Retrieved May 14 , 2011 . Jump up ^ Gibbs , Nancy ( November 13 , 2008 )
. `` How Presidents Pass The Torch '' . Time . Retrieved May 6 , 2011 . Jump up ^ Dorning , Mike ( January 22 , 2009 ) . `` A note from Bush starts morning in
the Oval Office '' . Chicago Tribune . Archived from the original on December 28 , 2011 . Retrieved May 6 , 2011 .
</item 9>

<item 10>
President of the United States
Educational site on the American presidency `` Presidents ' Occupations '' . Retrieved August 20 , 2007 . Listing of every president 's occupations before and
after becoming the commander in chief `` The Masonic Presidents Tour '' . The Masonic Library and Museum of Pennsylvania . Archived from the original on October
12 , 2014 . Retrieved October 7 , 2005 . Brief histories of the Masonic careers of presidents who were members of the Freemasons `` The Presidents '' . American
Experience . Retrieved March 4 , 2007 . PBS site on the American presidency Presidents of the United States : Resource Guides from the Library of Congress
Shapell Manuscript Foundation . Images of documents written by U.S.
</item 10>
=== END ITEM id=aad4df820b6cc8df ===

=== ITEM id=ef5c38223324eceb ===
Gold sentences and facts:
Sentence 0: The Battle of Lake Erie , sometimes called the Battle of Put - in - Bay , was fought on 10 September 1813 , in Lake Erie off the coast of Ohio during the War of 1812 . Nine vessels of the United States Navy defeated and captured six vessels of the British Royal Navy . This ensured American control of the lake for the rest of the war , which in turn allowed the Americans to recover Detroit and win the Battle of the Thames to break the Indian confederation of Tecumseh . It was one of the biggest naval battles of the War of 1812 .
  Fact 0.0: The Battle of Lake Erie is sometimes called the Battle of Put-in-Bay.
  Fact 0.1: The Battle of Lake Erie was fought on 10 September 1813.
  Fact 0.2: The Battle of Lake Erie was fought in Lake Erie off the coast of Ohio.
  Fact 0.3: The Battle of Lake Erie was fought during the War of 1812.
  Fact 0.4: Nine vessels of the United States Navy defeated six vessels of the British Royal Navy in the Battle of Lake Erie.
  Fact 0.5: Nine vessels of the United States Navy captured six vessels of the British Royal Navy in the Battle of Lake Erie.
  Fact 0.6: The Battle of Lake Erie ensured American control of the lake for the rest of the War of 1812.
  Fact 0.7: The Battle of Lake Erie allowed the Americans to recover Detroit.
  Fact 0.8: The Battle of Lake Erie allowed the Americans to win the Battle of the Thames.
  Fact 0.9: Winning the Battle of the Thames broke the Indian confederation of Tecumseh.
  Fact 0.10: The Battle of Lake Erie was one of the biggest naval battles of the War of 1812.

Retrieved items:
<item 1>
Battle of Lake Erie
It was one of the biggest naval battles of the War of 1812 . Contents ( hide ) 1 Background 1.1 1812 1.2 1813 1.3 Blockades of Presque Isle and Amherstburg 2
Battle 3 Casualties 4 Aftermath 5 Reconstructions and memorials 6 Reasons for the American victory 7 Order of battle 8 Footnotes 9 Notes 10 References 11
Further reading 12 External links Background ( edit ) 1812 ( edit ) When the war broke out , the British immediately seized control of Lake Erie . They already
had a small force of warships there : the sloop - of - war Queen Charlotte and the brig General Hunter . The schooner Lady Prevost was under construction and
was put into service a few weeks after the outbreak of war . These vessels were controlled by the Provincial Marine , which was a military transport service and
not a naval service . Nevertheless , the Americans lacked any counter to the British armed vessels .
</item 1>

<item 2>
Battle of Lake Erie
The weather was hazy , and he missed them . During July and August , Barclay received two small vessels , the schooner Chippeway and the sloop Little Belt ,
which had been reconstructed at Chatham on the Thames River and attempted to complete the ship - rigged corvette HMS Detroit at Amherstburg . Because the
Americans controlled Lake Ontario and occupied the Niagara Peninsula in early 1813 , supplies for Barclay had to be carried overland from York . The American
victory earlier in the year at the Battle of York resulted in the guns ( 24 - pounder carronades ) intended for Detroit falling into American hands . Detroit
had to be completed with a miscellany of guns from the fortifications of Amherstburg . Barclay claimed at his court martial that these guns lacked flintlock
firing mechanisms and matches , and that they could be fired only by snapping flintlock pistols over powder piled in the vent holes . Barclay repeatedly
requested men and supplies from Commodore James Lucas Yeo , commanding on Lake Ontario , but received very little .
</item 2>

<item 3>
Battle of Lake Erie
James Massacre Missouri Battle of Credit Island Battle of the Sink Hole British North America Lower Canada Battle of the Chateauguay First Battle of Lacolle
Mills Second Battle of Lacolle Mills Upper Canada Battle of Beaver Dams Battle of Chippawa Battle of Cook 's Mills Battle of Crysler 's Farm Battle of Fort
George Battle of Frenchman 's Creek Battle of Longwoods Battle of Lundy 's Lane Battle of Malcolm 's Mills Battle of Queenston Heights Battle of Stoney Creek
Battle of the Thames Battle of York Capture of Fort Erie Raid on Elizabethtown Raid on Port Dover Raid on Gananoque Siege of Fort Erie Spanish Empire Spanish
Florida Battle of Pensacola Naval battles Atlantic Ocean Capture of HMS Boxer Capture of HMS Cyane Capture of HMS Epervier Capture of HMS Frolic Capture of HMS
Penguin Capture of HMS Dominica Capture of USS Argus Capture of USS Chesapeake Capture of USS President Chesapeake Bay Flotilla USS Constitution vs HMS Java
Sinking of HMS Avon Battle of Fayal Sinking of HMS Peacock Sinking of HMS Reindeer USS Constitution vs HMS Guerriere USS United States vs HMS Macedonian
Caribbean Sea Battle of La Guaira Great Lakes Battle of Lake Erie
</item 3>

<item 4>
Battle of Lake Erie
Michaels Raid on Havre de Grace Massachusetts Battle of Hampden New York Battle of Big Sandy Creek Battle of Buffalo Battle of Ogdensburg Battle of Plattsburgh
Capture of Fort Niagara Raid on Black Rock Second Battle of Sacket 's Harbor Ohio Battle of Fort Stephenson Copus massacre Siege of Fort Meigs Virginia Battle
of Craney Island Raid on Alexandria Skirmish at Farnham Church Battle of Rappahannock River U.S. territories Alabama Battles of Fort Bowyer Illinois Battle of
Fort Dearborn Battle of Rock Island Rapids Siege of Prairie du Chien Indiana Battle of the Mississinewa Battle of Tippecanoe Battle of Wild Cat Creek Siege of
Fort Harrison Siege of Fort Wayne Michigan Battle of Brownstown Battle of Frenchtown Battle of Mackinac Island Battle of Maguaga Siege of Detroit Siege of Fort
Mackinac Mississippi Battle of Burnt Corn Battle of Callabee Creek Canoe Fight Battle of Holy Ground Battle of Horseshoe Bend Battle of Talladega Battle of
Tallushatchee Battles of Emuckfaw and Enotachopo Creek Fort Mims massacre Kimbell -- James Massacre Missouri Battle of Credit Island Battle of the Sink Hole
British North America Lower Canada Battle of
</item 4>

<item 5>
Battle of Lake Erie
Also present at Black Rock were the schooners Somers and Ohio and the sloop - rigged Trippe , which had all been purchased by the United States Navy and were
being converted into gunboats . While the British held Fort Erie and the nearby batteries which dominated the Niagara River , all these vessels were pinned down
and unable to leave Black Rock . Late in 1812 , Paul Hamilton , the United States Secretary of the Navy had received long - time American lake mariner Daniel
Dobbins , who had escaped capture at Detroit and brought information on the British forces on Lake Erie . Dobbins recommended the bay of Presque Isle in Erie ,
Pennsylvania as a naval base on the lake . ( `` Presqu'isle '' is French for `` peninsula '' , literally `` almost an island '' ) . Dobbins was dispatched to
build four gunboats there , although Lieutenant Elliot objected to the lack of facilities . Commodore Isaac Chauncey had been appointed to command of the United
States naval forces on the Great Lakes in September 1812 .
</item 5>

<item 6>
Battle of Lake Erie
Naval campaigns of the War of 1812 Atlantic Ocean USS Essex vs HMS Alert USS Constitution vs HMS Guerriere Capture of HMS Frolic USS United States vs HMS
Macedonian USS Constitution vs HMS Java Sinking of HMS Peacock Rappahannock River Capture of USS Chesapeake Capture of HMS Dominica Capture of USS Argus Capture
of HMS Boxer Capture of USS Frolic Capture of HMS Epervier Sinking of HMS Reindeer Sinking of HMS Avon Fayal Capture of USS President Capture of HMS Cyane and
HMS Levant Capture of HMS Penguin Capture of East India Company ship Nautilus East Coast Chesapeake Bay Alexandria Baltimore Hampden Fort Peter Great Lakes /
Saint Lawrence River Lake Ontario 1st Sacket 's Harbor York Fort George 2nd Sacket 's Harbor Lake Erie Fort Oswego Lake Huron Lake Champlain West Indies / Gulf
Coast La Guaira 1st Fort Bowyer Action of 13 December 1814 Lake Borgne New Orleans Fort St.
</item 6>

<item 7>
Battle of Lake Erie
Fort Bowyer Action of 13 December 1814 Lake Borgne New Orleans Fort St. Philip 2nd Fort Bowyer Pacific Ocean James Island Charles Island Nuku Hiva Downes
Expedition Porter Expedition Typee Valley Valparaiso ( Capture of USS Essex ) Seringapatam Mutiny Action of 9 May 1814 The Battle of Lake Erie , sometimes
called the Battle of Put - in - Bay , was fought on 10 September 1813 , in Lake Erie off the coast of Ohio during the War of 1812 . Nine vessels of the United
States Navy defeated and captured six vessels of the British Royal Navy . This ensured American control of the lake for the rest of the war , which in turn
allowed the Americans to recover Detroit and win the Battle of the Thames to break the Indian confederation of Tecumseh . It was one of the biggest naval
battles of the War of 1812 .
</item 7>

<item 8>
Battle of Lake Erie
USS United States vs HMS Macedonian Caribbean Sea Battle of La Guaira Great Lakes Battle of Lake Erie Battle of Fort Oswego Engagements on Lake Huron
Engagements on Lake Ontario First Battle of Sacket 's Harbor Gulf Coast Action of 13 December 1814 Battle of Lake Borgne Pacific Ocean Action off James Island
Action off Charles Island Nuku Hiva Campaign Battle of Valparaiso ( Capture of USS Essex ) See also : American Indian Wars , Creek War , Napoleonic Wars , and
Tecumseh 's War Portal definition textbooks quotes source texts media news stories Coordinates : 41 ° 39 ′ 44 '' N 82 ° 49 ′ 30 '' W ﻿ / ﻿ 41.66222 ° N 82.82500
° W ﻿ / 41.66222 ; - 82.82500 Retrieved from `` https://en.wikipedia.org/w/index.
</item 8>

<item 9>
Battle of Lake Erie
( Four were destroyed when the British captured Black Rock , where they were laid up , following the Battle of Buffalo at the end of 1813 , and four were
boarded and captured in separate incidents on Lake Erie and Lake Huron . ) After the war , there was a bitter quarrel between Perry and Elliot over their
respective parts in the action , mostly fought at second hand in the press . On the British side , Barclay was exonerated of any blame by a court - martial but
was too badly injured to see service again for several years . Reconstructions and memorials ( edit ) In 1820 Lawrence and Niagara were intentionally sunk near
Misery Bay in Lake Erie , as they had `` went to rot . '' In 1875 , Lawrence was raised and moved to Philadelphia , where she was displayed at the 1876
Centennial Exposition . Later that year , the ship burned when the pavilion that housed it caught fire . Although Niagara was raised and restored in 1913 , she
subsequently fell into disrepair . She was eventually disassembled , and portions of her were used in a reconstructed Niagara , which is now on view in Erie ,
Pennsylvania .
</item 9>

<item 10>
Battle of Lake Erie
Battle of Lake Erie - wikipedia Battle of Lake Erie Jump to : navigation , search Battle of Lake Erie Part of the War of 1812 Battle of Lake Erie by William
Henry Powell , painted 1865 , shows Oliver Hazard Perry transferring from Lawrence to Niagara Date 10 September 1813 Location Lake Erie , near Put - in - Bay ,
Ohio Result Decisive American victory Belligerents United States United Kingdom Commanders and leaders Oliver Hazard Perry Jesse Elliot Robert Heriot Barclay
Robert Finnis Strength 5 schooners 3 brigs 1 sloop 2 ships 1 brig 2 schooners 1 sloop Casualties and losses 27 killed 96 wounded 1 brig severely damaged 41
killed 93 wounded 306 captured entire squadron captured Detroit frontier Tippecanoe Fort Mackinac ( 1812 ) Brownstown Maguaga Fort Dearborn Detroit Fort
Harrison Fort Wayne Wild Cat Creek Mississinewa Frenchtown Africa Point Fort Meigs Fort Stephenson Lake Erie Thames Longwoods Prairie du Chien Campbell Island
Mackinac Island ( 1814 ) Lake Huron Malcolm 's Mills Naval campaigns of the War of 1812 Atlantic Ocean USS Essex vs HMS Alert USS Constitution vs HMS
</item 10>
=== END ITEM id=ef5c38223324eceb ===
