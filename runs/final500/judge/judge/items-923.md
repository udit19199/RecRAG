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

=== ITEM id=e071080560cf37a7 ===
Gold sentences and facts:
Sentence 0: No . overall No. in season Title Directed by Written by Original air date Prod . code U.S. viewers ( millions ) 107 `` Watch Me '' Jann Turner Shonda Rhimes October 5 , 2017 ( 2017 - 10 - 05 ) 701 5.52 108 `` Pressing the Flesh '' Tony Goldwyn Matt Byrne October 12 , 2017 ( 2017 - 10 - 12 ) 702 5.00 109 `` Day 101 '' Scott Foley Zahir McGhee October 19 , 2017 ( 2017 - 10 - 19 ) 703 4.70 110 `` Lost Girls '' Nicole Rubio Ameni Rozsa & Austin Guzman October 26 , 2017 ( 2017 - 10 - 26 ) 704 4.88 111 5 `` Adventures in Babysitting '' Oliver Bokelberg Serveriano Canales & Tia Napolitano November 2 , 2017 ( 2017 - 11 - 02 ) 705 4.89 112 6 `` Vampires and Bloodsuckers '' Jann Turner Chris Van Dusen & Tia Napolitano November 9 , 2017 ( 2017 - 11 - 09 ) 706 5.00 113 7 `` Something Borrowed '' Sharat Raju Mark Fish November 16 , 2017 ( 2017 - 11 - 16 ) 707 4.97 114 8 `` Robin '' Daryn Okada Juan Carlos Fernandez January 18 , 2018 ( 2018 - 01 - 18 ) 709 5.17 115 9 `` Good People '' Nzingha Stewart Shonda Rhimes , Jess Brownell & Nicholas Nardini January 25 , 2018 ( 2018 - 01 - 25 ) 708 TBD 116 10 `` The People v. Olivia Pope '' TBA TBA February 1 , 2018 ( 2018 - 02 - 01 ) 710 TBD 117 11 `` Army of One '' TBA TBA February 8 , 2018 ( 2018 - 02 - 08 ) 711 TBD 118 12 TBA TBA TBA March 1 , 2018 ( 2018 - 03 - 01 ) TBA TBD
  Fact 0.0: Episode 107 overall is titled 'Watch Me'.
  Fact 0.1: Episode 107 overall was directed by Jann Turner.
  Fact 0.2: Episode 107 overall was written by Shonda Rhimes.
  Fact 0.3: Episode 107 overall originally aired on October 5, 2017.
  Fact 0.4: Episode 107 overall has production code 701.
  Fact 0.5: Episode 107 overall had 5.52 million U.S. viewers.
  Fact 0.6: Episode 108 overall is titled 'Pressing the Flesh'.
  Fact 0.7: Episode 108 overall was directed by Tony Goldwyn.
  Fact 0.8: Episode 108 overall was written by Matt Byrne.
  Fact 0.9: Episode 108 overall originally aired on October 12, 2017.
  Fact 0.10: Episode 108 overall has production code 702.
  Fact 0.11: Episode 108 overall had 5.00 million U.S. viewers.
  Fact 0.12: Episode 109 overall is titled 'Day 101'.
  Fact 0.13: Episode 109 overall was directed by Scott Foley.
  Fact 0.14: Episode 109 overall was written by Zahir McGhee.
  Fact 0.15: Episode 109 overall originally aired on October 19, 2017.
  Fact 0.16: Episode 109 overall has production code 703.
  Fact 0.17: Episode 109 overall had 4.70 million U.S. viewers.
  Fact 0.18: Episode 110 overall is titled 'Lost Girls'.
  Fact 0.19: Episode 110 overall was directed by Nicole Rubio.
  Fact 0.20: Episode 110 overall was written by Ameni Rozsa and Austin Guzman.
  Fact 0.21: Episode 110 overall originally aired on October 26, 2017.
  Fact 0.22: Episode 110 overall has production code 704.
  Fact 0.23: Episode 110 overall had 4.88 million U.S. viewers.
  Fact 0.24: Episode 111 overall is episode 5 of its season and is titled 'Adventures in Babysitting'.
  Fact 0.25: Episode 111 overall was directed by Oliver Bokelberg.
  Fact 0.26: Episode 111 overall was written by Serveriano Canales and Tia Napolitano.
  Fact 0.27: Episode 111 overall originally aired on November 2, 2017.
  Fact 0.28: Episode 111 overall has production code 705.
  Fact 0.29: Episode 111 overall had 4.89 million U.S. viewers.
  Fact 0.30: Episode 112 overall is episode 6 of its season and is titled 'Vampires and Bloodsuckers'.
  Fact 0.31: Episode 112 overall was directed by Jann Turner.
  Fact 0.32: Episode 112 overall was written by Chris Van Dusen and Tia Napolitano.
  Fact 0.33: Episode 112 overall originally aired on November 9, 2017.
  Fact 0.34: Episode 112 overall has production code 706.
  Fact 0.35: Episode 112 overall had 5.00 million U.S. viewers.
  Fact 0.36: Episode 113 overall is episode 7 of its season and is titled 'Something Borrowed'.
  Fact 0.37: Episode 113 overall was directed by Sharat Raju.
  Fact 0.38: Episode 113 overall was written by Mark Fish.
  Fact 0.39: Episode 113 overall originally aired on November 16, 2017.
  Fact 0.40: Episode 113 overall has production code 707.
  Fact 0.41: Episode 113 overall had 4.97 million U.S. viewers.
  Fact 0.42: Episode 114 overall is episode 8 of its season and is titled 'Robin'.
  Fact 0.43: Episode 114 overall was directed by Daryn Okada.
  Fact 0.44: Episode 114 overall was written by Juan Carlos Fernandez.
  Fact 0.45: Episode 114 overall originally aired on January 18, 2018.
  Fact 0.46: Episode 114 overall has production code 709.
  Fact 0.47: Episode 114 overall had 5.17 million U.S. viewers.
  Fact 0.48: Episode 115 overall is episode 9 of its season and is titled 'Good People'.
  Fact 0.49: Episode 115 overall was directed by Nzingha Stewart.
  Fact 0.50: Episode 115 overall was written by Shonda Rhimes, Jess Brownell and Nicholas Nardini.
  Fact 0.51: Episode 115 overall originally aired on January 25, 2018.
  Fact 0.52: Episode 115 overall has production code 708.
  Fact 0.53: Episode 115 overall has U.S. viewership listed as TBD.
  Fact 0.54: Episode 116 overall is episode 10 of its season and is titled 'The People v. Olivia Pope'.
  Fact 0.55: Episode 116 overall has its director listed as TBA.
  Fact 0.56: Episode 116 overall has its writer listed as TBA.
  Fact 0.57: Episode 116 overall originally aired on February 1, 2018.
  Fact 0.58: Episode 116 overall has production code 710.
  Fact 0.59: Episode 116 overall has U.S. viewership listed as TBD.
  Fact 0.60: Episode 117 overall is episode 11 of its season and is titled 'Army of One'.
  Fact 0.61: Episode 117 overall has its director listed as TBA.
  Fact 0.62: Episode 117 overall has its writer listed as TBA.
  Fact 0.63: Episode 117 overall originally aired on February 8, 2018.
  Fact 0.64: Episode 117 overall has production code 711.
  Fact 0.65: Episode 117 overall has U.S. viewership listed as TBD.
  Fact 0.66: Episode 118 overall is episode 12 of its season.
  Fact 0.67: Episode 118 overall has its title, director, and writer listed as TBA.
  Fact 0.68: Episode 118 overall originally aired on March 1, 2018.
  Fact 0.69: Episode 118 overall has its production code listed as TBA.
  Fact 0.70: Episode 118 overall has U.S. viewership listed as TBD.
Sentence 1: 116 10 `` The People v. Olivia Pope '' TBA TBA February 1 , 2018 ( 2018 - 02 - 01 ) 710 TBD
  Fact 1.0: Episode 116 overall is episode 10 of its season and is titled 'The People v. Olivia Pope'.
  Fact 1.1: Episode 116 overall has its director listed as TBA.
  Fact 1.2: Episode 116 overall has its writer listed as TBA.
  Fact 1.3: Episode 116 overall originally aired on February 1, 2018.
  Fact 1.4: Episode 116 overall has production code 710.
  Fact 1.5: Episode 116 overall has U.S. viewership listed as TBD.

Retrieved items:
<item 1>
List of Scandal episodes
2017 ( 2017 - 03 - 16 ) 605 5.63 97 7 `` A Traitor Among Us '' Tom Verica Alison Schapker March 23 , 2017 ( 2017 - 03 - 23 ) 607 5.40 98 8 `` A Stomach for
Blood '' Oliver Bokelberg Severiano Canales March 30 , 2017 ( 2017 - 03 - 30 ) 608 6.57 99 9 `` Dead in the Water '' Nicole Rubio Michelle Lirtzman April 6 ,
2017 ( 2017 - 04 - 06 ) 609 5.10 100 10 `` The Decision '' Sharat Raju Johanna Lee April 13 , 2017 ( 2017 - 04 - 13 ) 610 5.35 101 11 `` Trojan Horse '' Jann
Turner Jess Brownell & Nicholas Nardini April 20 , 2017 ( 2017 - 04 - 20 ) 611 5.
</item 1>

<item 2>
List of Scandal episodes
2014 ( 2014 - 10 - 16 ) 404A 9.90 52 5 `` The Key '' Paul McCrane Chris Van Dusen October 23 , 2014 ( 2014 - 10 - 23 ) 404B 9.98 53 6 `` An Innocent Man ''
Jeannot Szwarc Zahir McGhee October 30 , 2014 ( 2014 - 10 - 30 ) 405 9.32 54 7 `` Baby Made a Mess '' Oliver Bokelberg Jenna Bans November 6 , 2014 ( 2014 - 11
- 06 ) 406 9.82 55 8 `` The Last Supper '' Julie Anne Robinson Allan Heinberg November 13 , 2014 ( 2014 - 11 - 13 ) 407 10.05 56 9 `` Where the Sun Do n't Shine
'' Tony Goldwyn Mark Wilding November 20 , 2014 ( 2014 - 11 - 20 ) 408 10.14 57 10 `` Run '' Tom Verica Shonda Rhimes January 29 ,
</item 2>

<item 3>
List of Scandal episodes
2015 ( 2015 - 10 - 01 ) 502 9.12 72 `` Paris is Burning '' Jann Turner Matt Byrne October 8 , 2015 ( 2015 - 10 - 08 ) 503 8.76 73 `` Dog - Whistle Politics ''
Zetna Fuentes Mark Fish October 15 , 2015 ( 2015 - 10 - 15 ) 504 8.06 74 5 `` You Got Served '' Kevin Bray Zahir McGhee October 22 , 2015 ( 2015 - 10 - 22 ) 505
8.28 75 6 `` Get Out of Jail , Free '' Chandra Wilson Chris Van Dusen October 29 , 2015 ( 2015 - 10 - 29 ) 506 7.80 76 7 `` Even the Devil Deserves a Second
Chance '' Oliver Bokelberg Raamla Mohamed November 5 , 2015 ( 2015 - 11 - 05 ) 507 8.03 77 8 `` Rasputin '' John Terlesky Paul William Davies November 12 ,
</item 3>

<item 4>
List of Scandal episodes
2013 ( 2013 - 03 - 21 ) 216 8.51 24 17 `` Snake in the Garden '' Ron Underwood Raamla Mohamed March 28 , 2013 ( 2013 - 03 - 28 ) 217 7.97 25 18 `` Molly , You
in Danger , Girl '' Tom Verica Chris Van Dusen April 4 , 2013 ( 2013 - 04 - 04 ) 218 8.04 26 19 `` Seven Fifty - Two '' Allison Liddi - Brown Mark Fish April 25
, 2013 ( 2013 - 04 - 25 ) 219 7.90 27 20 `` A Woman Scorned '' Tony Goldwyn Zahir McGhee May 2 , 2013 ( 2013 - 05 - 02 ) 220 8.07 28 21 `` Any Questions ? ''
</item 4>

<item 5>
List of Scandal episodes
2014 ( 2014 - 03 - 13 ) 313 8.22 43 14 `` Kiss Kiss Bang Bang '' Paul McCrane Mark Fish March 20 , 2014 ( 2014 - 03 - 20 ) 314 9.08 44 15 `` Mama Said Knock You
Out '' Tony Goldwyn Zahir McGhee March 27 , 2014 ( 2014 - 03 - 27 ) 315 9.01 45 16 `` The Fluffer '' Jeannot Szwarc Chris Van Dusen & Raamla Mohamed April 3 ,
2014 ( 2014 - 04 - 03 ) 316 9.13 46 17 `` Flesh and Blood '' Debbie Allen Severiano Canales & Miguel Nolla April 10 , 2014 ( 2014 - 04 - 10 ) 317 9.23 47 18 ``
The Price of Free and Fair Elections '' Tom Verica Shonda Rhimes & Mark Wilding April 17 , 2014 ( 2014 - 04 - 17 ) 318 10.
</item 5>

<item 6>
List of Scandal episodes
2017 ( 2017 - 10 - 26 ) 704 4.88 111 5 `` Adventures in Babysitting '' Oliver Bokelberg Serveriano Canales & Tia Napolitano November 2 , 2017 ( 2017 - 11 - 02 )
705 4.89 112 6 `` Vampires and Bloodsuckers '' Jann Turner Chris Van Dusen & Tia Napolitano November 9 , 2017 ( 2017 - 11 - 09 ) 706 5.00 113 7 `` Something
Borrowed '' Sharat Raju Mark Fish November 16 , 2017 ( 2017 - 11 - 16 ) 707 4.97 114 8 `` Robin '' Daryn Okada Juan Carlos Fernandez January 18 , 2018 ( 2018 -
01 - 18 ) 709 5.17 115 9 `` Good People '' Nzingha Stewart Shonda Rhimes , Jess Brownell & Nicholas Nardini January 25 ,
</item 6>

<item 7>
List of Scandal episodes
2012 ( 2012 - 11 - 08 ) 205 6.06 13 6 `` Spies Like Us '' Bethany Rooney Chris Van Dusen November 15 , 2012 ( 2012 - 11 - 15 ) 206 6.02 14 7 `` Defiance '' Tom
Verica Peter Noah November 29 , 2012 ( 2012 - 11 - 29 ) 207 6.64 15 8 `` Happy Birthday , Mr. President '' Oliver Bokelberg Shonda Rhimes December 6 , 2012 (
2012 - 12 - 06 ) 208 7.39 16 9 `` Blown Away '' Jessica Yu Mark Wilding December 13 , 2012 ( 2012 - 12 - 13 ) 209 7.14 17 10 `` One for the Dog '' Steve Robin
Heather Mitchell January 10 , 2013 ( 2013 - 01 - 10 ) 210 8.37 18 11 `` A Criminal , a Whore ,
</item 7>

<item 8>
List of Scandal episodes
2017 ( 2017 - 04 - 20 ) 611 5.11 102 12 `` Mercy '' Nzingha Stewart Severiano Canales & Ameni Rozsa April 27 , 2017 ( 2017 - 04 - 27 ) 612 5.29 103 13 `` The
Box '' Steph Green Raamla Mohamed & Austin Guzman May 4 , 2017 ( 2017 - 05 - 04 ) 613 5.15 104 14 `` Head Games '' Zetna Fuentes Chris Van Dusen & Juan Carlos
Fernandez May 11 , 2017 ( 2017 - 05 - 11 ) 614 5.10 105 15 `` Tick Tock '' Salli Richardson - Whitfield Zahir McGhee & Michelle Lirtzman May 18 , 2017 ( 2017 -
05 - 18 ) 615 5.23 106 16 `` Transfer of Power '' Tony Goldwyn Matt Byrne & Mark Fish May 18 , 2017 ( 2017 - 05 - 18 ) 616 5.
</item 8>

<item 9>
List of Scandal episodes
2016 ( 2016 - 03 - 10 ) 512 5.97 83 14 `` I See You '' Paris Barclay Matt Byrne March 17 , 2016 ( 2016 - 03 - 17 ) 513 6.31 84 15 `` Pencils Down '' Regina King
Chris Van Dusen March 24 , 2016 ( 2016 - 03 - 24 ) 514 6.15 85 16 `` The Miseducation of Susan Ross '' Scott Foley Raamla Mohamed March 31 , 2016 ( 2016 - 03 -
31 ) 516 6.44 86 17 `` Thwack ! ''
</item 9>

<item 10>
List of Scandal episodes
37 18 11 `` A Criminal , a Whore , an Idiot and a Liar '' Stephen Cragg Mark Fish January 17 , 2013 ( 2013 - 01 - 17 ) 211 7.93 19 12 `` Truth or Consequences
'' Jeannot Szwarc Peter Noah January 31 , 2013 ( 2013 - 01 - 31 ) 212 8.09 20 13 `` Nobody Likes Babies '' Tom Verica Mark Wilding February 7 , 2013 ( 2013 - 02
- 07 ) 213 8.14 21 14 `` Whiskey Tango Foxtrot '' Mark Tinker Matt Byrne February 14 , 2013 ( 2013 - 02 - 14 ) 214 8.02 22 15 `` Boom Goes the Dynamite '' Randy
Zisk Jenna Bans February 21 , 2013 ( 2013 - 02 - 21 ) 215 7.68 23 16 `` Top of the Hour '' Steve Robin Heather Mitchell March 21 , 2013 ( 2013 - 03 - 21 ) 216
8.
</item 10>
=== END ITEM id=e071080560cf37a7 ===

=== ITEM id=be70e9e1c89c3642 ===
Gold sentences and facts:
Sentence 0: Jane and Rafael start saving up to buy a one - bedroom apartment , but their plan falls apart when Mateo breaks an antique $4,000 clock . Rafael moves into the Villanueva house , but after Alba hits Mateo , the two get into a heated argument and Rafael moves back out . Xiomara starts chemotherapy , and makes a friend named Donna . The timeline moves forward nine weeks , and Donna dies of unforeseen consequences . Xiomara is shown not to be doing well due to the chemotherapy , Alba and Rafael are still in a fight and River and Rogelio are still arguing over the Passions of Santos adaptation . Alba breaks down to Jane , and she and Rafael make up . Xiomara bonds with River over telenovelas , after she accidentally eats a pot brownie , and she agrees to do the Passions of Steve and Brenda the Latin American way . Jane Ramos gets herself disbarred , and as a result gets the case against Petra thrown out .
  Fact 0.0: Jane and Rafael start saving up to buy a one-bedroom apartment.
  Fact 0.1: Jane and Rafael's plan to buy the apartment falls apart when Mateo breaks an antique $4,000 clock.
  Fact 0.2: Rafael moves into the Villanueva house.
  Fact 0.3: After Alba hits Mateo, Alba and Rafael get into a heated argument.
  Fact 0.4: Rafael moves back out of the Villanueva house after the argument.
  Fact 0.5: Xiomara starts chemotherapy.
  Fact 0.6: Xiomara makes a friend named Donna.
  Fact 0.7: The timeline moves forward nine weeks.
  Fact 0.8: Donna dies of unforeseen consequences.
  Fact 0.9: Xiomara is shown not to be doing well due to the chemotherapy.
  Fact 0.10: Alba and Rafael are still in a fight.
  Fact 0.11: River and Rogelio are still arguing over the Passions of Santos adaptation.
  Fact 0.12: Alba breaks down to Jane.
  Fact 0.13: Alba and Rafael make up.
  Fact 0.14: Xiomara bonds with River over telenovelas.
  Fact 0.15: Xiomara accidentally eats a pot brownie.
  Fact 0.16: Xiomara agrees to do the Passions of Steve and Brenda the Latin American way.
  Fact 0.17: Jane Ramos gets herself disbarred.
  Fact 0.18: As a result of Jane Ramos being disbarred, the case against Petra is thrown out.
Sentence 1: No . overall No. in season Title Directed by Written by Original air date U.S. viewers ( millions ) 65 `` Chapter Sixty - Five '' Brad Silberling Jennie Snyder Urman & Paul Sciarrotta October 13 , 2017 ( 2017 - 10 - 13 ) 0.68 Picking up immediately after where Season 3 left off , Jane is reunited with her first love , Adam , who found the love letter that Michael wrote . She is surprised , but happy , to see him . However , both Xo and Alba are less thrilled that he is back . Meanwhile , Rafael loses ownership of the hotel to his sister and as a result , he finds himself living at Jane 's house , broke and homeless . Jane and Rafael learn that the missing Petra has been found after she got away from Anezka and they rush to her side , but Anezka is nowhere to be found . When Petra returns to the hotel , Luisa thinks that Petra is Anezka . Petra goes along with the game , until the real Anezka returns . Elsewhere , Xo and Rogelio 's married life is interrupted when they confront the realization that Darci is having Rogelio 's baby and she wants full custody . 66 `` Chapter Sixty - Six '' Gina Lamar Valentina Garza & Jessica O'Toole & Amy Rardin October 20 , 2017 ( 2017 - 10 - 20 ) 0.61 Jane 's relationship with Adam helps bring out her `` fun side '' and reminds her that she is n't just a mom . However , when Adam gets a job opportunity out of state , he asks for Jane 's opinion on if he should stay or go . Meanwhile , Rogelio continues to face issues with his co-star at work and things are about to get even worse . Xo is helping to take care of Darci to prove that Rogelio is committed to being a father to the baby . Also , Petra and Rafael begin working together to try and buy back the hotel from Luisa , but they are running into trouble involving Anezka , causing tension between them . 67 `` Chapter Sixty - Seven '' Fernando Sariñana Carolina Rivera & Micah Schraft October 27 , 2017 ( 2017 - 10 - 27 ) 0.60 Jane and Adam 's relationship seems to be moving on , and Jane thinks that it might be time for him to meet her son Mateo , but Rafael is not happy with this idea . Meanwhile , as the arrival of Darci 's baby approaches , Rogelio 's jealously of her new boyfriend increases as well . Trying to be supportive , Xo encourages Rogelio to step up and be the father he was meant to be . Elsewhere , Rafael is determined to get his hotel and life back from Luisa and will do whatever he has to do to make it happen , with or without Petra 's help . 68 `` Chapter Sixty - Eight '' Gina Lamar Deirdre Shaw & Chantelle M. Wells November 3 , 2017 ( 2017 - 11 - 03 ) 0.69 Jane becomes angry when she learns that Rafael introduced Mateo to his new girlfriend , Katherine , without her consent because she thinks Katherine is a negative influence on both Rafael and Mateo . Jane turns to Petra for help with the situation . Meanwhile , Rogelio and Darci struggle to choose a name for the baby . Elsewhere , Xo , who is focused on opening her own dance studio , decides to go into business with an unlikely foe . 69 5 `` Chapter Sixty - Nine '' Stuart Gillard Paul Sciarrotta November 10 , 2017 ( 2017 - 11 - 10 ) 0.65 After Adam reveals his dating past to Jane , she becomes so unsure and insecure that it puts a strain on their relationship . Meanwhile , Lina seeks Jane 's advice about whether she should get married or not . Petra hears some important information that could help Rafael get the hotel back . Also , Xo asks Rogelio to consider having a medical procedure ( vasectomy ) done , but he is very hesitant . 70 6 `` Chapter Seventy '' Melanie Mayron Carolina Rivera & Micah Schraft November 17 , 2017 ( 2017 - 11 - 17 ) 0.61 Jane 's book is finally being released but her plans to have a party at the local bookshop where she was inspired to be a writer hit a snag . Always wanting to help Jane , Rogelio devises a plan to save the day , but one of his arch - rivals threatens to ruin the day . Meanwhile , Rafael gets a new job , but it is not exactly what he was hoping for . Also , Petra has come up with an interesting strategy to get the Marbella Hotel back from Luisa , but Rafael wants nothing to do with the scheme . 71 7 `` Chapter Seventy - One '' Micah Schraft Valentina Garza & Deidre Shaw December 8 , 2017 ( 2017 - 12 - 08 ) 0.65 Anezka is revealed to have faked her death to see where her loyalties lie ; initially , she trusts Petra after believing she saw her cry , but after finding out she was n't crying because of her fake death , she threatens Petra . Later , Anezka falls from Petra 's apartment to her death ... for real . Meanwhile , Xiomara and Jane set off on Jane 's book tour , when Alba decides to tag along . After they get back , Jane and Rafael share a kiss . 72 8 `` Chapter Seventy - Two '' Melanie Mayron Carolina Rivera & Paul Sciarrotta January 26 , 2018 ( 2018 - 01 - 26 ) 0.68 Mateo starts kindergarten and invites a friend over to Xiomara and Rogelio 's house . Jane and Rafael try to cover up that they are pretending to live in their house so they do n't get kicked out . Jane and Rafael show mixed emotions following the kiss and try again , which Mateo notices . He then proclaims : `` Does this mean you 're going to get married ? '' After Xo convinces Rogelio to attend couples therapy , they both are surprised by their emotions . Meanwhile , Petra hires a lawyer named Jane Ramos ( Rosario Dawson ) , to ensure she gets no charges pending Anezka 's death . Also , Rafael moves out of Jane 's house into his new apartment . 73 9 `` Chapter Seventy - Three '' Eric Lea Valentina Garza & Chantelle M. Wells February 2 , 2018 ( 2018 - 02 - 02 ) 0.67 When Jane and Rafael finally realize that the uncertain state of their relationship ( or lack of it ) is stressing out Mateo , they decide that they need to make a decision . Meanwhile , Petra is on the outs with Jane and Rafael after the Luisa fiasco , so she tries to figure out a way to get back into their good graces . Elsewhere , Rogelio and Xo are both contemplating their career choices . 74 10 `` Chapter Seventy - Four '' Gina Rodriguez Micah Schraft & Paul Sciarrotta February 9 , 2018 ( 2018 - 02 - 09 ) 0.80 Jane and Rafael try to be friends , which works great until Jane goes on a date . Meanwhile , Petra and Jane 's new partnership is off to a rocky start , but they both know they have to find a way to work together somehow . Rogelio is convinced he has male post-partum depression , which puts him at odds with with celebrity Post-Partum Advocate River Fields ( guest star Brooke Shields ) on The Talk . Also , Alba finally confides in Jane why she turned down the marriage proposal , leaving Jane to help Alba in the romance department when Jane takes Alba to a sex store and buys her a vibrator . Elsewhere , Jane and Rafael agree to have sex , and Petra questions her own sexual identity when she starts to have sex dreams about Jane `` J.R. '' Ramos after the lawyer begins flirting with her . 75 11 `` Chapter Seventy - Five '' Gina Lamar Merigan Mulhern March 2 , 2018 ( 2018 - 03 - 02 ) 0.59 Jane is ready to move on from the Marbella Hotel issues and pursue writing full time , but an unexpected development forces her to consider returning to her teaching roots . Xiomara decides to go back to school where she listens in on an art history class , while Jane teaches a seminar about writing . Here , Jane finds out Jonathan Chavez , her former advisor , is taking advantage of another student of his . While trying to warn the student , Jane accidentally grabs her breasts to avoid falling over and , as a result , does n't get the teaching job she was applying for . Meanwhile , Petra gets arrested for Anezka 's murder due to the blackmailer planting the screws in her apartment , leading Jane Ramos to tell Petra that she was framing her because she was being threatened by an unknown blackmailer . Also , Rafael and Jane continue their affair . Elsewhere , Rogelio tries to get Eva Longoria to join his Passions of Santos reboot , but instead finds out the two are distant cousins . 76 12 `` Chapter Seventy - Six '' Melanie Mayron Leah Longoria March 9 , 2018 ( 2018 - 03 - 09 ) 0.58 Jane 's fixation on a negative review has left her with writer 's block , so Rogelio suggests she take a writing class to help . With the encouragement of Jane , Rafael is ready to dig into his past to find out who his real biological parents are . Meanwhile , Rogelio is trying to make things less about him and more about Xo , but he grows suspicious when he learns she is keeping a secret from him . But the secret Xo has been keeping is not what Rogelio suspected . Elsewhere , Petra tests the loyalty of her lawyer , Jane Ramos , by pretending to be romantically interested in her while trying to find out the identity of the blackmailer . After Petra confides in Jane R. of her feelings which are not fake , the two of them sleep together for the first time . 77 13 `` Chapter Seventy - Seven '' Gina Lamar Deirdre Shaw & Chantelle M. Wells March 16 , 2018 ( 2018 - 03 - 16 ) 0.59 When Jane , Rogelio and Alba learn what is going on with Xo who might have breast cancer , they all try to comfort her , but Xo insists they let her live her life . Meanwhile , Rogelio finally gets River Fields to agree to meet with him , but he learns that River has another plan in mind . Also , Rafael is on a mission to find his sister , who may have the information he has been looking for of his origins . Rafael visits Rose in prison to ask for her help , but he quickly turns her down when she makes a demand to know Luisa 's location . Also , Jane is upset with Petra after her twin girls spill a secret to Mateo about the Tooth Fairy being make believe . At the same time , Jane ' J.R. ' Ramos tells Petra after their romantic night together that she is not interested in a relationship due to the blackmailer paying a visit to Jane R. 's dementia - stricken mother in a rest home . At the end , the identity of Jane R 's blackmailer is finally revealed . 78 14 `` Chapter Seventy - Eight '' Justin Baldoni Valentina Garza & Micah Schraft March 23 , 2018 ( 2018 - 03 - 23 ) 0.65 Xo struggles to make an important decision on whether or not to get a full mastectomy and looks to Jane for guidance , leaving Rogelio feeling like an outsider . Alba is frustrated with Rogelio when it seems he is once again making his career a priority over Xo 's well being . Meanwhile , Petra turns to Rafael for advice on her love life after he finally learns about her new romance with J.R. 79 15 `` Chapter Seventy - Nine '' Gina Lamar Paul Sciarrotta April 6 , 2018 ( 2018 - 04 - 06 ) 0.53 80 16 `` Chapter Eighty '' Micah Schraft Micah Schraft April 13 , 2018 ( 2018 - 04 - 13 ) 0.57 Jane and Rafael start saving up to buy a one - bedroom apartment , but their plan falls apart when Mateo breaks an antique $4,000 clock . Rafael moves into the Villanueva house , but after Alba hits Mateo , the two get into a heated argument and Rafael moves back out . Xiomara starts chemotherapy , and makes a friend named Donna . The timeline moves forward nine weeks , and Donna dies of unforeseen consequences . Xiomara is shown not to be doing well due to the chemotherapy , Alba and Rafael are still in a fight and River and Rogelio are still arguing over the Passions of Santos adaptation . Alba breaks down to Jane , and she and Rafael make up . Xiomara bonds with River over telenovelas , after she accidentally eats a pot brownie , and she agrees to do the Passions of Steve and Brenda the Latin American way . Jane Ramos gets herself disbarred , and as a result gets the case against Petra thrown out . 81 17 `` Chapter Eighty - One '' Gina Lamar Jennie Snyder Urman & Paul Sciarrotta April 20 , 2018 ( 2018 - 04 - 20 ) 0.58 Rafael intends to propose to Jane , however Rose 's news unsettles him . Alba becomes a citizen and the Villanueva 's throw her a surprise party . Jane Ramos and Petra plan to move in together , until Jane finds out that Petra killed Anezka . As she walks out , Krishna tells her that her life is in danger . J.R. saves Petra 's life by shooting , and killing , the blackmailer . As Jane meets Rafael , Michael is revealed to be alive .
  Fact 1.0: The listing has columns for overall episode number, season episode number, title, director, writers, original air date, and U.S. viewers in millions.
  Fact 1.1: Chapter Sixty-Five is episode 65 overall, directed by Brad Silberling and written by Jennie Snyder Urman and Paul Sciarrotta.
  Fact 1.2: Chapter Sixty-Five first aired October 13, 2017, with 0.68 million U.S. viewers.
  Fact 1.3: Chapter Sixty-Five picks up immediately after where Season 3 left off.
  Fact 1.4: Jane is reunited with her first love, Adam, who found the love letter that Michael wrote.
  Fact 1.5: Jane is surprised but happy to see Adam.
  Fact 1.6: Xo and Alba are less thrilled that Adam is back.
  Fact 1.7: Rafael loses ownership of the hotel to his sister.
  Fact 1.8: As a result, Rafael finds himself living at Jane's house, broke and homeless.
  Fact 1.9: Jane and Rafael learn that the missing Petra has been found after she got away from Anezka.
  Fact 1.10: Jane and Rafael rush to Petra's side, but Anezka is nowhere to be found.
  Fact 1.11: When Petra returns to the hotel, Luisa thinks Petra is Anezka.
  Fact 1.12: Petra goes along with the game until the real Anezka returns.
  Fact 1.13: Xo and Rogelio's married life is interrupted when they confront the realization that Darci is having Rogelio's baby.
  Fact 1.14: Darci wants full custody of the baby.
  Fact 1.15: Chapter Sixty-Six is directed by Gina Lamar and written by Valentina Garza, Jessica O'Toole and Amy Rardin.
  Fact 1.16: Chapter Sixty-Six first aired October 20, 2017, with 0.61 million U.S. viewers.
  Fact 1.17: Jane's relationship with Adam helps bring out her 'fun side'.
  Fact 1.18: Adam's relationship with Jane reminds her that she is not just a mom.
  Fact 1.19: Adam gets a job opportunity out of state.
  Fact 1.20: Adam asks Jane's opinion on whether he should stay or go.
  Fact 1.21: Rogelio continues to face issues with his co-star at work.
  Fact 1.22: Rogelio's issues with his co-star are about to get even worse.
  Fact 1.23: Xo is helping to take care of Darci to prove that Rogelio is committed to being a father to the baby.
  Fact 1.24: Petra and Rafael begin working together to try to buy back the hotel from Luisa.
  Fact 1.25: Petra and Rafael run into trouble involving Anezka.
  Fact 1.26: Petra and Rafael's trouble involving Anezka causes tension between them.
  Fact 1.27: Chapter Sixty-Seven is directed by Fernando Sariñana and written by Carolina Rivera and Micah Schraft.
  Fact 1.28: Chapter Sixty-Seven first aired October 27, 2017, with 0.60 million U.S. viewers.
  Fact 1.29: Jane and Adam's relationship seems to be moving on.
  Fact 1.30: Jane thinks it might be time for Adam to meet her son Mateo.
  Fact 1.31: Rafael is not happy with the idea of Adam meeting Mateo.
  Fact 1.32: As the arrival of Darci's baby approaches, Rogelio's jealousy of Darci's new boyfriend increases.
  Fact 1.33: Xo encourages Rogelio to step up and be the father he was meant to be.
  Fact 1.34: Rafael is determined to get his hotel and life back from Luisa.
  Fact 1.35: Rafael will do whatever he has to do to get his hotel and life back, with or without Petra's help.
  Fact 1.36: Chapter Sixty-Eight is directed by Gina Lamar and written by Deirdre Shaw and Chantelle M. Wells.
  Fact 1.37: Chapter Sixty-Eight first aired November 3, 2017, with 0.69 million U.S. viewers.
  Fact 1.38: Jane becomes angry when she learns Rafael introduced Mateo to his new girlfriend, Katherine, without her consent.
  Fact 1.39: Jane thinks Katherine is a negative influence on both Rafael and Mateo.
  Fact 1.40: Jane turns to Petra for help with the situation.
  Fact 1.41: Rogelio and Darci struggle to choose a name for the baby.
  Fact 1.42: Xo is focused on opening her own dance studio.
  Fact 1.43: Xo decides to go into business with an unlikely foe.
  Fact 1.44: Chapter Sixty-Nine is episode 69 overall and season 5, directed by Stuart Gillard and written by Paul Sciarrotta.
  Fact 1.45: Chapter Sixty-Nine first aired November 10, 2017, with 0.65 million U.S. viewers.
  Fact 1.46: After Adam reveals his dating past to Jane, Jane becomes unsure and insecure.
  Fact 1.47: Jane's insecurity puts a strain on her relationship with Adam.
  Fact 1.48: Lina seeks Jane's advice about whether she should get married.
  Fact 1.49: Petra hears important information that could help Rafael get the hotel back.
  Fact 1.50: Xo asks Rogelio to consider having a vasectomy.
  Fact 1.51: Rogelio is very hesitant about the vasectomy.
  Fact 1.52: Chapter Seventy is episode 70 overall and season 6, directed by Melanie Mayron and written by Carolina Rivera and Micah Schraft.
  Fact 1.53: Chapter Seventy first aired November 17, 2017, with 0.61 million U.S. viewers.
  Fact 1.54: Jane's book is finally being released.
  Fact 1.55: Jane's plans to have a party at the local bookshop where she was inspired to be a writer hit a snag.
  Fact 1.56: Rogelio devises a plan to save the day for Jane.
  Fact 1.57: One of Rogelio's arch-rivals threatens to ruin the day.
  Fact 1.58: Rafael gets a new job, but it is not exactly what he was hoping for.
  Fact 1.59: Petra has come up with an interesting strategy to get the Marbella Hotel back from Luisa.
  Fact 1.60: Rafael wants nothing to do with Petra's scheme.
  Fact 1.61: Chapter Seventy-One is episode 71 overall and season 7, directed by Micah Schraft and written by Valentina Garza and Deidre Shaw.
  Fact 1.62: Chapter Seventy-One first aired December 8, 2017, with 0.65 million U.S. viewers.
  Fact 1.63: Anezka is revealed to have faked her death to see where her loyalties lie.
  Fact 1.64: Anezka initially trusts Petra after believing she saw Petra cry.
  Fact 1.65: Anezka threatens Petra after finding out Petra was not crying because of Anezka's fake death.
  Fact 1.66: Later, Anezka falls from Petra's apartment to her death, for real.
  Fact 1.67: Xiomara and Jane set off on Jane's book tour.
  Fact 1.68: Alba decides to tag along on Jane's book tour.
  Fact 1.69: After they get back, Jane and Rafael share a kiss.
  Fact 1.70: Chapter Seventy-Two is episode 72 overall and season 8, directed by Melanie Mayron and written by Carolina Rivera and Paul Sciarrotta.
  Fact 1.71: Chapter Seventy-Two first aired January 26, 2018, with 0.68 million U.S. viewers.
  Fact 1.72: Mateo starts kindergarten.
  Fact 1.73: Mateo invites a friend over to Xiomara and Rogelio's house.
  Fact 1.74: Jane and Rafael try to cover up that they are pretending to live in their house so they don't get kicked out.
  Fact 1.75: Jane and Rafael show mixed emotions following the kiss and try again, which Mateo notices.
  Fact 1.76: Mateo proclaims: 'Does this mean you're going to get married?'
  Fact 1.77: After Xo convinces Rogelio to attend couples therapy, Xo and Rogelio are both surprised by their emotions.
  Fact 1.78: Petra hires a lawyer named Jane Ramos, played by Rosario Dawson.
  Fact 1.79: Petra hires Jane Ramos to ensure Petra gets no charges pending Anezka's death.
  Fact 1.80: Rafael moves out of Jane's house into his new apartment.
  Fact 1.81: Chapter Seventy-Three is episode 73 overall and season 9, directed by Eric Lea and written by Valentina Garza and Chantelle M. Wells.
  Fact 1.82: Chapter Seventy-Three first aired February 2, 2018, with 0.67 million U.S. viewers.
  Fact 1.83: Jane and Rafael realize that the uncertain state of their relationship is stressing out Mateo.
  Fact 1.84: Jane and Rafael decide that they need to make a decision.
  Fact 1.85: Petra is on the outs with Jane and Rafael after the Luisa fiasco.
  Fact 1.86: Petra tries to figure out a way to get back into Jane and Rafael's good graces.
  Fact 1.87: Rogelio and Xo are both contemplating their career choices.
  Fact 1.88: Chapter Seventy-Four is episode 74 overall and season 10, directed by Gina Rodriguez and written by Micah Schraft and Paul Sciarrotta.
  Fact 1.89: Chapter Seventy-Four first aired February 9, 2018, with 0.80 million U.S. viewers.
  Fact 1.90: Jane and Rafael try to be friends, which works great until Jane goes on a date.
  Fact 1.91: Petra and Jane's new partnership is off to a rocky start.
  Fact 1.92: Petra and Jane both know they have to find a way to work together.
  Fact 1.93: Rogelio is convinced he has male post-partum depression.
  Fact 1.94: Rogelio's belief puts him at odds with celebrity Post-Partum Advocate River Fields on The Talk.
  Fact 1.95: River Fields is guest starred by Brooke Shields.
  Fact 1.96: Alba finally confides in Jane why she turned down the marriage proposal.
  Fact 1.97: Jane helps Alba in the romance department.
  Fact 1.98: Jane takes Alba to a sex store and buys her a vibrator.
  Fact 1.99: Jane and Rafael agree to have sex.
  Fact 1.100: Petra questions her own sexual identity.
  Fact 1.101: Petra starts to have sex dreams about Jane 'J.R.' Ramos after the lawyer begins flirting with her.
  Fact 1.102: Chapter Seventy-Five is episode 75 overall and season 11, directed by Gina Lamar and written by Merigan Mulhern.
  Fact 1.103: Chapter Seventy-Five first aired March 2, 2018, with 0.59 million U.S. viewers.
  Fact 1.104: Jane is ready to move on from the Marbella Hotel issues and pursue writing full time.
  Fact 1.105: An unexpected development forces Jane to consider returning to her teaching roots.
  Fact 1.106: Xiomara decides to go back to school, where she listens in on an art history class.
  Fact 1.107: Jane teaches a seminar about writing.
  Fact 1.108: Jane finds out Jonathan Chavez, her former advisor, is taking advantage of another student.
  Fact 1.109: While trying to warn the student, Jane accidentally grabs her breasts to avoid falling over.
  Fact 1.110: Because of this, Jane does not get the teaching job she was applying for.
  Fact 1.111: Petra gets arrested for Anezka's murder due to the blackmailer planting the screws in Petra's apartment.
  Fact 1.112: Jane Ramos tells Petra that she was framing Petra because Jane was being threatened by an unknown blackmailer.
  Fact 1.113: Rafael and Jane continue their affair.
  Fact 1.114: Rogelio tries to get Eva Longoria to join his Passions of Santos reboot.
  Fact 1.115: Rogelio finds out that he and Eva Longoria are distant cousins.
  Fact 1.116: Chapter Seventy-Six is episode 76 overall and season 12, directed by Melanie Mayron and written by Leah Longoria.
  Fact 1.117: Chapter Seventy-Six first aired March 9, 2018, with 0.58 million U.S. viewers.
  Fact 1.118: Jane's fixation on a negative review has left her with writer's block.
  Fact 1.119: Rogelio suggests Jane take a writing class to help.
  Fact 1.120: With Jane's encouragement, Rafael is ready to dig into his past to find out who his real biological parents are.
  Fact 1.121: Rogelio is trying to make things less about him and more about Xo.
  Fact 1.122: Rogelio grows suspicious when he learns Xo is keeping a secret from him.
  Fact 1.123: The secret Xo has been keeping is not what Rogelio suspected.
  Fact 1.124: Petra tests the loyalty of her lawyer, Jane Ramos, by pretending to be romantically interested in her.
  Fact 1.125: Petra is trying to find the identity of the blackmailer.
  Fact 1.126: Petra confides in Jane R. about her feelings, which are not fake.
  Fact 1.127: Petra and Jane R. sleep together for the first time.
  Fact 1.128: Chapter Seventy-Seven is episode 77 overall and season 13, directed by Gina Lamar and written by Deirdre Shaw and Chantelle M. Wells.
  Fact 1.129: Chapter Seventy-Seven first aired March 16, 2018, with 0.59 million U.S. viewers.
  Fact 1.130: Jane, Rogelio and Alba learn that Xo might have breast cancer.
  Fact 1.131: Jane, Rogelio and Alba try to comfort Xo.
  Fact 1.132: Xo insists they let her live her life.
  Fact 1.133: Rogelio finally gets River Fields to agree to meet with him.
  Fact 1.134: Rogelio learns that River has another plan in mind.
  Fact 1.135: Rafael is on a mission to find his sister, who may have information he has been looking for about his origins.
  Fact 1.136: Rafael visits Rose in prison to ask for her help.
  Fact 1.137: Rafael quickly turns Rose down when she demands to know Luisa's location.
  Fact 1.138: Jane is upset with Petra after Jane's twin girls spill a secret to Mateo about the Tooth Fairy being make-believe.
  Fact 1.139: Jane R. Ramos tells Petra she is not interested in a relationship after their romantic night together.
  Fact 1.140: The blackmailer paid a visit to Jane R.'s dementia-stricken mother in a rest home.
  Fact 1.141: The identity of Jane R.'s blackmailer is finally revealed at the end of the episode.
  Fact 1.142: Chapter Seventy-Eight is episode 78 overall and season 14, directed by Justin Baldoni and written by Valentina Garza and Micah Schraft.
  Fact 1.143: Chapter Seventy-Eight first aired March 23, 2018, with 0.65 million U.S. viewers.
  Fact 1.144: Xo struggles to make an important decision on whether or not to get a full mastectomy.
  Fact 1.145: Xo looks to Jane for guidance on the mastectomy decision.
  Fact 1.146: Xo's decision leaves Rogelio feeling like an outsider.
  Fact 1.147: Alba is frustrated with Rogelio when he once again seems to make his career a priority over Xo's well-being.
  Fact 1.148: Petra turns to Rafael for advice on her love life.
  Fact 1.149: Rafael finally learns about Petra's new romance with J.R.
  Fact 1.150: Chapter Seventy-Nine is episode 79 overall and season 15, directed by Gina Lamar and written by Paul Sciarrotta.
  Fact 1.151: Chapter Seventy-Nine first aired April 6, 2018, with 0.53 million U.S. viewers.
  Fact 1.152: Chapter Eighty is episode 80 overall and season 16, directed by Micah Schraft and written by Micah Schraft.
  Fact 1.153: Chapter Eighty first aired April 13, 2018, with 0.57 million U.S. viewers.
  Fact 1.154: Jane and Rafael start saving up to buy a one-bedroom apartment.
  Fact 1.155: Jane and Rafael's plan falls apart when Mateo breaks an antique $4,000 clock.
  Fact 1.156: Rafael moves into the Villanueva house.
  Fact 1.157: After Alba hits Mateo, Alba and Rafael get into a heated argument.
  Fact 1.158: Rafael moves back out after the argument.
  Fact 1.159: Xiomara starts chemotherapy.
  Fact 1.160: Xiomara makes a friend named Donna.
  Fact 1.161: The timeline moves forward nine weeks.
  Fact 1.162: Donna dies of unforeseen consequences.
  Fact 1.163: Xiomara is shown not to be doing well due to the chemotherapy.
  Fact 1.164: Alba and Rafael are still in a fight.
  Fact 1.165: River and Rogelio are still arguing over the Passions of Santos adaptation.
  Fact 1.166: Alba breaks down to Jane.
  Fact 1.167: Alba and Rafael make up.
  Fact 1.168: Xiomara bonds with River over telenovelas.
  Fact 1.169: Xiomara accidentally eats a pot brownie.
  Fact 1.170: Xiomara agrees to do the Passions of Steve and Brenda the Latin American way.
  Fact 1.171: Jane Ramos gets herself disbarred.
  Fact 1.172: As a result, Jane Ramos gets the case against Petra thrown out.
  Fact 1.173: Chapter Eighty-One is episode 81 overall and season 17, directed by Gina Lamar and written by Jennie Snyder Urman and Paul Sciarrotta.
  Fact 1.174: Chapter Eighty-One first aired April 20, 2018, with 0.58 million U.S. viewers.
  Fact 1.175: Rafael intends to propose to Jane.
  Fact 1.176: Rose's news unsettles Rafael.
  Fact 1.177: Alba becomes a citizen.
  Fact 1.178: The Villanueva family throws Alba a surprise party.
  Fact 1.179: Jane Ramos and Petra plan to move in together.
  Fact 1.180: Jane finds out Petra killed Anezka.
  Fact 1.181: As Jane walks out, Krishna tells her that her life is in danger.
  Fact 1.182: J.R. saves Petra's life by shooting and killing the blackmailer.
  Fact 1.183: As Jane meets Rafael, Michael is revealed to be alive.

Retrieved items:
<item 1>
Jane the Virgin (season 4)
Also , Rafael is on a mission to find his sister , who may have the information he has been looking for of his origins . Rafael visits Rose in prison to ask for
her help , but he quickly turns her down when she makes a demand to know Luisa 's location . Also , Jane is upset with Petra after her twin girls spill a secret
to Mateo about the Tooth Fairy being make believe . At the same time , Jane ' J.R. ' Ramos tells Petra after their romantic night together that she is not
interested in a relationship due to the blackmailer paying a visit to Jane R. 's dementia - stricken mother in a rest home . At the end , the identity of Jane R
's blackmailer is finally revealed . 78 14 `` Chapter Seventy - Eight '' Justin Baldoni Valentina Garza & Micah Schraft March 23 , 2018 ( 2018 - 03 - 23 ) 0.65
Xo struggles to make an important decision on whether or not to get a full mastectomy and looks to Jane for guidance , leaving Rogelio feeling like an outsider
.
</item 1>

<item 2>
Jane the Virgin (season 4)
However , both Xo and Alba are less thrilled that he is back . Meanwhile , Rafael loses ownership of the hotel to his sister and as a result , he finds himself
living at Jane 's house , broke and homeless . Jane and Rafael learn that the missing Petra has been found after she got away from Anezka and they rush to her
side , but Anezka is nowhere to be found . When Petra returns to the hotel , Luisa thinks that Petra is Anezka . Petra goes along with the game , until the real
Anezka returns . Elsewhere , Xo and Rogelio 's married life is interrupted when they confront the realization that Darci is having Rogelio 's baby and she wants
full custody . 66 `` Chapter Sixty - Six '' Gina Lamar Valentina Garza & Jessica O'Toole & Amy Rardin October 20 , 2017 ( 2017 - 10 - 20 ) 0.61 Jane 's
relationship with Adam helps bring out her `` fun side '' and reminds her that she is n't just a mom .
</item 2>

<item 3>
Jane the Virgin (season 4)
However , when Adam gets a job opportunity out of state , he asks for Jane 's opinion on if he should stay or go . Meanwhile , Rogelio continues to face issues
with his co-star at work and things are about to get even worse . Xo is helping to take care of Darci to prove that Rogelio is committed to being a father to
the baby . Also , Petra and Rafael begin working together to try and buy back the hotel from Luisa , but they are running into trouble involving Anezka ,
causing tension between them . 67 `` Chapter Sixty - Seven '' Fernando Sariñana Carolina Rivera & Micah Schraft October 27 , 2017 ( 2017 - 10 - 27 ) 0.60 Jane
and Adam 's relationship seems to be moving on , and Jane thinks that it might be time for him to meet her son Mateo , but Rafael is not happy with this idea .
Meanwhile , as the arrival of Darci 's baby approaches , Rogelio 's jealously of her new boyfriend increases as well .
</item 3>

<item 4>
Jane the Virgin (season 4)
Meanwhile , Rogelio is trying to make things less about him and more about Xo , but he grows suspicious when he learns she is keeping a secret from him . But
the secret Xo has been keeping is not what Rogelio suspected . Elsewhere , Petra tests the loyalty of her lawyer , Jane Ramos , by pretending to be romantically
interested in her while trying to find out the identity of the blackmailer . After Petra confides in Jane R. of her feelings which are not fake , the two of
them sleep together for the first time . 77 13 `` Chapter Seventy - Seven '' Gina Lamar Deirdre Shaw & Chantelle M. Wells March 16 , 2018 ( 2018 - 03 - 16 )
0.59 When Jane , Rogelio and Alba learn what is going on with Xo who might have breast cancer , they all try to comfort her , but Xo insists they let her live
her life . Meanwhile , Rogelio finally gets River Fields to agree to meet with him , but he learns that River has another plan in mind .
</item 4>

<item 5>
Jane the Virgin (season 4)
Also , Rafael moves out of Jane 's house into his new apartment . 73 9 `` Chapter Seventy - Three '' Eric Lea Valentina Garza & Chantelle M. Wells February 2 ,
2018 ( 2018 - 02 - 02 ) 0.67 When Jane and Rafael finally realize that the uncertain state of their relationship ( or lack of it ) is stressing out Mateo , they
decide that they need to make a decision . Meanwhile , Petra is on the outs with Jane and Rafael after the Luisa fiasco , so she tries to figure out a way to
get back into their good graces . Elsewhere , Rogelio and Xo are both contemplating their career choices . 74 10 `` Chapter Seventy - Four '' Gina Rodriguez
Micah Schraft & Paul Sciarrotta February 9 , 2018 ( 2018 - 02 - 09 ) 0.80 Jane and Rafael try to be friends , which works great until Jane goes on a date .
Meanwhile , Petra and Jane 's new partnership is off to a rocky start , but they both know they have to find a way to work together somehow .
</item 5>

<item 6>
Jane the Virgin (season 4)
Trying to be supportive , Xo encourages Rogelio to step up and be the father he was meant to be . Elsewhere , Rafael is determined to get his hotel and life
back from Luisa and will do whatever he has to do to make it happen , with or without Petra 's help . 68 `` Chapter Sixty - Eight '' Gina Lamar Deirdre Shaw &
Chantelle M. Wells November 3 , 2017 ( 2017 - 11 - 03 ) 0.69 Jane becomes angry when she learns that Rafael introduced Mateo to his new girlfriend , Katherine ,
without her consent because she thinks Katherine is a negative influence on both Rafael and Mateo . Jane turns to Petra for help with the situation . Meanwhile
, Rogelio and Darci struggle to choose a name for the baby . Elsewhere , Xo , who is focused on opening her own dance studio , decides to go into business with
an unlikely foe .
</item 6>

<item 7>
Jane the Virgin (season 4)
After they get back , Jane and Rafael share a kiss . 72 8 `` Chapter Seventy - Two '' Melanie Mayron Carolina Rivera & Paul Sciarrotta January 26 , 2018 ( 2018
- 01 - 26 ) 0.68 Mateo starts kindergarten and invites a friend over to Xiomara and Rogelio 's house . Jane and Rafael try to cover up that they are pretending
to live in their house so they do n't get kicked out . Jane and Rafael show mixed emotions following the kiss and try again , which Mateo notices . He then
proclaims : `` Does this mean you 're going to get married ? '' After Xo convinces Rogelio to attend couples therapy , they both are surprised by their emotions
. Meanwhile , Petra hires a lawyer named Jane Ramos ( Rosario Dawson ) , to ensure she gets no charges pending Anezka 's death . Also , Rafael moves out of Jane
's house into his new apartment .
</item 7>

<item 8>
Jane the Virgin (season 4)
69 5 `` Chapter Sixty - Nine '' Stuart Gillard Paul Sciarrotta November 10 , 2017 ( 2017 - 11 - 10 ) 0.65 After Adam reveals his dating past to Jane , she
becomes so unsure and insecure that it puts a strain on their relationship . Meanwhile , Lina seeks Jane 's advice about whether she should get married or not .
Petra hears some important information that could help Rafael get the hotel back . Also , Xo asks Rogelio to consider having a medical procedure ( vasectomy )
done , but he is very hesitant . 70 6 `` Chapter Seventy '' Melanie Mayron Carolina Rivera & Micah Schraft November 17 , 2017 ( 2017 - 11 - 17 ) 0.61 Jane 's
book is finally being released but her plans to have a party at the local bookshop where she was inspired to be a writer hit a snag . Always wanting to help
Jane , Rogelio devises a plan to save the day , but one of his arch - rivals threatens to ruin the day .
</item 8>

<item 9>
Jane the Virgin (season 4)
Meanwhile , Rafael gets a new job , but it is not exactly what he was hoping for . Also , Petra has come up with an interesting strategy to get the Marbella
Hotel back from Luisa , but Rafael wants nothing to do with the scheme . 71 7 `` Chapter Seventy - One '' Micah Schraft Valentina Garza & Deidre Shaw December 8
, 2017 ( 2017 - 12 - 08 ) 0.65 Anezka is revealed to have faked her death to see where her loyalties lie ; initially , she trusts Petra after believing she saw
her cry , but after finding out she was n't crying because of her fake death , she threatens Petra . Later , Anezka falls from Petra 's apartment to her death
... for real . Meanwhile , Xiomara and Jane set off on Jane 's book tour , when Alba decides to tag along . After they get back , Jane and Rafael share a kiss .
</item 9>

<item 10>
Jane the Virgin (season 4)
Alba is frustrated with Rogelio when it seems he is once again making his career a priority over Xo 's well being . Meanwhile , Petra turns to Rafael for advice
on her love life after he finally learns about her new romance with J.R. 79 15 `` Chapter Seventy - Nine '' Gina Lamar Paul Sciarrotta April 6 , 2018 ( 2018 -
04 - 06 ) 0.53 80 16 `` Chapter Eighty '' Micah Schraft Micah Schraft April 13 , 2018 ( 2018 - 04 - 13 ) 0.57 Jane and Rafael start saving up to buy a one -
bedroom apartment , but their plan falls apart when Mateo breaks an antique $4,000 clock . Rafael moves into the Villanueva house , but after Alba hits Mateo ,
the two get into a heated argument and Rafael moves back out . Xiomara starts chemotherapy , and makes a friend named Donna . The timeline moves forward nine
weeks , and Donna dies of unforeseen consequences .
</item 10>
=== END ITEM id=be70e9e1c89c3642 ===

=== ITEM id=b56120f6e3a9849a ===
Gold sentences and facts:
Sentence 0: At the federal level in the United States , legislation ( i.e. , `` statutes '' or `` statutory law '' ) consists exclusively of Acts passed by the Congress of the United States and its predecessor , the Continental Congress , that were either signed into law by the President or passed by Congress after a presidential veto .
  Fact 0.0: At the federal level in the United States, legislation consists exclusively of Acts.
  Fact 0.1: At the federal level in the United States, legislation is also called statutes or statutory law.
  Fact 0.2: Federal legislation in the United States consists exclusively of Acts passed by the Congress of the United States.
  Fact 0.3: Federal legislation in the United States consists exclusively of Acts passed by the Continental Congress, the predecessor of the Congress of the United States.
  Fact 0.4: Acts passed by the Congress of the United States or the Continental Congress were signed into law by the President.
  Fact 0.5: Acts passed by the Congress of the United States or the Continental Congress were passed by Congress after a presidential veto.

Retrieved items:
<item 1>
List of United States federal legislation
This numbering is continuous across sessions of Congress and does not include acts relating to individuals , which are analogously given Private Law X-Y
designations . ( From 1901 to 1957 , the Statutes at Large listed both a chapter and a Public / Private number for each act ) See examples below. There are many
laws . Examples ( edit ) The Militia Act of 1862 of July 17 , 1862 , Sess. 2 , ch. 201 , was the 201st Act of the second session of the 37th Congress . The
National Banking Act of February 25 , 1863 , Sess. 3 , ch. 58 , was the 58th Act of the third session of the 37th Congress . The Global Anti-Semitism Review Act
of 2004 of October 16 , 2004 , Pub. L. 108 -- 332 , 118 Stat. 1282 , was the 332nd Act of Congress ( statute ) passed in the 108th Congress . It can be found in
volume 118 of the U.S. Statutes at Large , starting at page 1282 .
</item 1>

<item 2>
State legislature (United States)
Constitution . Under Article II , state legislatures choose the manner of appointing the state 's presidential electors . Formerly , state legislatures
appointed the U.S. Senators from their respective states until the ratification of the 17th Amendment in 1913 required the direct election of Senators by the
state 's voters . Lawmaking process ( edit ) Generally , the legislative bodies and their committees use either Mason 's Manual of Legislative Procedure or an
amended form thereof . During official meetings , a professional parliamentarian is available to ensure that legislation and accompanying discussion proceed as
orderly as possible without bias . Beginnings ( edit ) The lawmaking process begins with the introduction of a bill in either the House of Representatives or
the Senate . Bills may be introduced in either house , sometimes with the exception of bills increasing or decreasing revenue , which must originate in the
House of Representatives . The order of business in each house provides a proper time for the introduction of bills . Bills are usually assigned consecutive
numbers , given in the order of their introduction , to facilitate identification . Usually a bill can not become enacted until it has been read on a certain
number of days in each house .
</item 2>

<item 3>
List of United States federal legislation
( edit ) Acts of Congress are published in the United States Statutes at Large . Volumes 1 through 18 , which have all the statutes passed from 1789 to 1875 ,
are available on - line at the Library of Congress , here . In the list below , statutes are listed by X Stat . Y , where X is the volume of the Statutes at
Large and Y is the page number , as well as either the chapter or Public Law number . See examples below . Sessions ( Sess. . ) and Chapters ( ch. . ) ( edit )
Each Congress has two to four sessions . Under the numbering system used from 1789 until 1957 , the Acts in each session are numbered sequentially as Chapters .
This numbering included both laws applicable to the general public and laws relating to specific individuals , e.g. , to grant pensions to disabled veterans .
Public laws ( edit ) Since 1957 , Acts of Congress related to the general public have been designated in the form : Public Law X-Y where X is the number of the
ordinal Congress and Y is the number of the chronological order of the public Act in that Congress .
</item 3>

<item 4>
List of United States federal legislation
However , most executive branch and Judicial Branch regulations must originate in a congressional grant of power . See also : Executive orders of the President
; regulations of Executive branch departments and administrative agencies ; and the procedural rules of the federal courts . Contents ( hide ) 1 Publication of
the law 1.1 Statutes at Large ( Stat . ) 1.2 Sessions ( Sess . ) and Chapters ( ch . ) 1.3 Public laws 1.4 Examples 2 Congress of the Confederation 3 United
States Congress 3.1 1789 to 1901 : 1st through 56th Congresses 3.2 1901 to 2001 : 57th through 106th Congresses 3.3 2001 to present : 107th and subsequent
Congresses 4 See also 5 Sources 6 References Publication of the law ( edit ) Statutes at Large ( Stat. . ) ( edit ) Acts of Congress are published in the United
States Statutes at Large .
</item 4>

<item 5>
State legislature (United States)
Conference committees ( edit ) A conference committee is often empaneled to discuss the points of difference between the two houses ' versions of the same bill
, and tries to reach an agreement between them so that the identical bill can be passed by both houses . If an agreement is reached and if both houses adopt the
conference committee 's report , the bill is passed . If either house refuses to adopt the report of the conference committee , a motion may be made for further
conference . If a conference committee is unable to reach an agreement , it may be discharged , and a new conference committee may be appointed . Some highly
controversial bills may be referred to several different conference committees . If an agreement is never reached in conference prior to the end of the
legislative session , the bill is lost . When a bill has passed both houses in identical form , it is then ready for transmittal to the Governor . Presentation
to the governor ( edit ) Once a bill reaches the governor , he or she may sign it , which completes its enactment into law . From this point , the bill becomes
an act , and remains the law of the state unless repealed by legislative action , or overturned by a court decision .
</item 5>

<item 6>
National Flood Insurance Program
National flood insurance program - wikipedia National flood insurance program Jump to : navigation , search Not to be confused with National Fire Protection
Association . `` NFIP '' redirects here . For the National Foundation for Infantile Paralysis , see March of Dimes . The National Flood Insurance Program ( NFIP
) is a program created by the Congress of the United States in 1968 through the National Flood Insurance Act of 1968 ( P.L. 90 - 448 ) . The program enables
property owners in participating communities to purchase insurance protection , administered by the government , against losses from flooding , and requires
flood insurance for all loans or lines of credit that are secured by existing buildings , manufactured homes , or buildings under construction , that are
located in a community that participates in the NFIP . This NFIP is designed to provide an insurance alternative to disaster assistance to meet the escalating
costs of repairing damage to buildings and their contents caused by floods . As of August 2017 , the program insured about 5 million homes ( down from about 5.5
million homes in April 2010 ) , the majority of which are in Texas and Florida .
</item 6>

<item 7>
State legislature (United States)
If the majority vote in favor of the bill , it is recorded as passed . Transmission to second house ( edit ) A bill that is passed in one house is transmitted ,
along with a formal message , to the other house . If the bill is not reported from committee or is not considered by the full house , the bill is defeated .
The house of origin , upon return of its amended bill , may take any one of several courses of action . It may concur in the amendment by the adoption of a
motion to that effect ; then the bill , having been passed by both houses in identical form , is ready for enrollment . Another possibility is that the house of
origin may adopt a motion to non-concur in the amendment , at which point the bill dies . Finally , the house of origin may refuse to accept the amendment but
request that a conference committee be appointed . The other house usually agrees to the request , and the presiding officer of each house appoints members to
the conference committee .
</item 7>

<item 8>
State legislature (United States)
If the governor does not approve of the bill , he or she may veto it . In the event of a veto , the governor returns the bill to the house in which it
originated , with a message explaining his or her objections and suggesting any amendments ( if applicable ) which might remove those objections . The bill is
then reconsidered , and if a simple majority of the members of both houses agrees to the proposed executive amendments , it is returned to the Governor , as he
or she revised it , for his or her signature . On the other hand , a simple majority of the members of each house can choose to approve a vetoed bill precisely
as the Legislature originally passed it , in which case it becomes a law over the governor 's veto . This is in contrast to the practice in most states and the
federal government , which require a two - thirds majority in both houses to override a governor 's veto . If the Governor fails to return a bill to the
legislative house in which it originated within a specified number of days after it was presented to him or her , it becomes a law without their signature .
</item 8>

<item 9>
List of United States federal legislation
Template : US tax acts , and Category : United States federal taxation legislation Transportation : United States Department of Transportation # Related
legislation Veterans Affairs : United States Department of Veterans Affairs # Related legislation Water Resources , Navigation , Environmental Regulation :
United States Army Corps of Engineers # Public Laws affecting the Corps of Engineers ( partial ) Sources ( edit ) Acts listed by popular name , via Cornell
University Statutes at Large Volumes 1 through 18 , 1789 -- 1875 , via Library of Congress Volumes 19 through 64 , 1875 - 1950 , via Library of Congress Volumes
65 through 125 , 1951 - 2011 , via Government Publishing Office Public laws 93rd Congress through current Congress , via Congress.gov U.S. Code U.S. Code , via
Law Revision Counsel of the U.S. House of Representatives U.S. Code , via Cornell University U.S. Code , via the U.S. Government Publishing Office U.S. Code ,
via FindLaw.com Brian K. Landsberg ( ed ) , Major Acts of Congress .
</item 9>

<item 10>
List of United States federal legislation
List of United States federal legislation - wikipedia List of United States federal legislation Jump to : navigation , search This article includes a list of
references , but its sources remain unclear because it has insufficient inline citations . Please help to improve this article by introducing more precise
citations. ( April 2016 ) ( Learn how and when to remove this template message ) See also : Category : United States federal legislation This is a chronological
, but still incomplete , list of United States federal legislation . Congress has enacted approximately 200 -- 600 statutes during each of its 115 biennial
terms , so that more than 20,000 statutes have been enacted since 1789 . At the federal level in the United States , legislation ( i.e. , `` statutes '' or ``
statutory law '' ) consists exclusively of Acts passed by the Congress of the United States and its predecessor , the Continental Congress , that were either
signed into law by the President or passed by Congress after a presidential veto . Legislation is not the only source of regulations with the force of law .
However , most executive branch and Judicial Branch regulations must originate in a congressional grant of power .
</item 10>
=== END ITEM id=b56120f6e3a9849a ===

=== ITEM id=24d3ebe079a70062 ===
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
Dutta 1993 -- 1999 85 -- Maulana Habibur Rahman Nomani 1993 -- 1999 86 Nirmala Deshpande 1997 -- 1999 and 2004 -- 2010 87 Shabana Azmi 1997 -- 2003 88 -- P.
Selvie Das 1997 -- 2003 89 -- Kartar Singh Duggal 1997 -- 2003 90 Kuldip Nayyar 1997 -- 2003 91 -- Raja Ramanna 1997 -- 2003 92 C.
</item 2>

<item 3>
List of nominated members of Rajya Sabha
Krishnamachari 1961 -- 1967 24 -- R.R. Diwakar 1962 -- 1968 25 -- Gopal Singh 1962 -- 1968 26 -- M. Ajmal Khan 1964 -- 1972 27 -- Satyavrata Siddhantalankar
1964 -- 1968 28 Shakuntala Paranjpye 1964 -- 1970 29 -- Badri Nath Prasad 1964 -- 1970 30 -- G. Ramachandran 1964 -- 1970 31 -- M.N. Kaul 1964 -- 1972 32 --
Harivanshrai Bachchan 1966 -- 1972 33 -- Dhananjay Ramchandra Gadgil 1966 -- 1972 34 -- M.C.
</item 3>

<item 4>
List of nominated members of Rajya Sabha
Sr. No . Image Name Tenure -- Alladi Krishnaswamy Iyer 1952 -- 1953 Satyendranath Bose 1952 -- 1960 Prithviraj Kapoor 1952 -- 1960 -- Jagadisan Mohandas
Kumarappa 1952 -- 1954 5 -- Kalidas Nag 1952 -- 1954 6 Rukmini Devi Arundale 1952 -- 1962 7 -- N.R. Malkani 1952 -- 1962 8 -- Sahib Singh Sokhey 1952 -- 1956 9
-- Zakir Hussain 1952 -- 1962 10 -- Maithili Sharan Gupt 1952 -- 1964 11 -- Kakasaheb Kalelkar 1952 -- 1964 12 -- Radha Kumud Mukherjee 1952 -- 1958 13 P.V.
</item 4>

<item 5>
List of nominated members of Rajya Sabha
Tiwari 1982 -- 1984 63 -- Ghulam Rasool Kar 1984 -- 1987 64 -- Tindivanam K. Ramamurthy 1984 -- 1990 65 -- H.L. Kapur 1985 -- 1988 66 Salim Ali 1985 -- 1988 67
-- Purushottam Kakodkar 1985 -- 1991 68 Ela Bhatt 1986 -- 1988 69 -- Amrita Pritam 1986 -- 1992 70 M.F. Husain 1986 -- 1992 71 R.K.
</item 5>

<item 6>
List of nominated members of Rajya Sabha
Husain 1986 -- 1992 71 R.K. Narayan 1986 -- 1992 72 Ravi Shankar 1986 -- 1992 73 -- Syeda Anwara Taimur 1988 -- 1990 74 -- Sat Paul Mittal 1988 -- 1994 75 --
Bishambhar Nath Pande 1988 -- 1994 76 -- Mohammad Yunus 1989 -- 1995 77 Prakash Ambedkar 1990 -- 1996 78 Jagmohan 1990 -- 1996 79 -- Bhupinder Singh Mann 1990
-- 1996 80 -- Russi Karanjia 1991 -- 1997 81 -- Mahendra Prasad 1993 -- 1994 82 -- M. Aram 1993 -- 1999 83 Vyjayantimala Bali 1993 -- 1999 84 -- B.B.
</item 6>

<item 7>
List of nominated members of Rajya Sabha
Daphtary 1972 -- 1978 47 -- Habib Tanvir 1972 -- 1978 48 -- Krishna Kriplani 1974 -- 1980 49 Lokesh Chandra 1974 -- 1986 50 -- Scato Swu 1974 -- 1986 51 -- B.N.
Banerjee 1976 -- 1982 52 -- Malcolm Adiseshiah 1978 -- 1984 53 -- Fathema Ismail 1978 -- 1984 54 -- Pandurang Dharmaji Jadhav 1978 -- 1984 55 -- Bhagwati Charan
Varma 1978 -- 1984 56 Nargis Dutt 1980 -- 1986 57 Khushwant Singh 1980 -- 1986 58 -- Asima Chatterjee 1982 -- 1990 59 Sivaji Ganesan 1982 -- 1986 60 --
Hayatullah Ansari 1982 -- 1988 61 -- Madan Bhatia 1982 -- 1994 62 -- V.N.
</item 7>

<item 8>
List of nominated members of Rajya Sabha
Narayana Reddy 1997 -- 2003 93 Mrinal Sen 1997 -- 2003 94 -- Chaudhary Harmohan Singh Yadav 1997 -- 2003 95 -- Nanaji Deshmukh 1999 -- 2005 96 Lata Mangeshkar
1999 -- 2005 97 -- Fali Nariman 1999 -- 2005 98 -- Cho Ramaswamy 1999 -- 2005 99 Hema Malini 2003 -- 2009 100 -- Bimal Jalan 2003 -- 2009 101 K.
</item 8>

<item 9>
List of nominated members of Rajya Sabha
No . Name Field Affiliation Date of Appointment Date of Retirement Roopa Ganguly Art Bharatiya Janata Party 04 - Oct - 2016 03 - Oct - 2022 Sambhaji Raje Social
work Bharatiya Janata Party 07 - Jun - 2016 03 - May - 2022 Suresh Gopi Art Bharatiya Janata Party 25 - Apr - 2016 24 - Apr - 2022 Subramanian Swamy Economics
Bharatiya Janata Party 25 - Apr - 2016 24 - Apr - 2022 5 Narendra Jadhav Economics Nominated 25 - Apr - 2016 24 - Apr - 2022 6 Mary Kom Sport Nominated 25 - Apr
- 2016 24 - Apr - 2022 7 Swapan Dasgupta Journalism Nominated 25 - Apr - 2016 24 - Apr - 2022 8 K.T.S.
</item 9>

<item 10>
List of nominated members of Rajya Sabha
Tulsi 2014 - 2020 125 Subramanian Swamy 2016 - 2022 126 MC Mary Kom 2016 - 2022 127 Narendra Jadhav 2016 - 2022 129 Suresh Gopi 2016 - 2022 130 -- Swapan
Dasgupta 2016 - 2022 131 -- Sambhaji Raje 2016 - 2022 132 -- Roopa Ganguly 2016 - 2022 References ( edit ) Nominated Members since 1952 Narender Kumar Current
list of Nominated Members Specific Jump up ^ `` Statewise List '' . 164.100. 47.5 . Retrieved 12 June 2016 . Jump up ^
http://164.100.47.5/Newmembers/partymemberlist.aspx ^ Jump up to : `` Statewise Retirement '' . 164.100. 47.5 . Retrieved 12 June 2016 .
</item 10>
=== END ITEM id=24d3ebe079a70062 ===

=== ITEM id=328d55daa0710995 ===
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
Number of deaths due to AIDS by country World Health Organisation ; Number of people ( all ages ) living with HIV by country ( show ) HIV / AIDS topics ( show )
HIV / AIDS HIV HIV structure and genome subtypes CDC classification disease progression rates HIV / AIDS diagnosis management pathophysiology prevention
research vaccination PrEP WHO disease staging system for HIV infection and disease Children Teens / Adults Countries by AIDS prevalence rate Conditions Signs
and symptoms AIDS - defining clinical condition Diffuse infiltrative lymphocytosis syndrome Lipodystrophy Nephropathy Neurocognitive disorders Pruritus
Superinfection Tuberculosis co-infection HIV Drug Resistance Database Innate resistance to HIV History History Epidemiology Multiple sex partners Timeline AIDS
Museum Timothy Ray Brown Social AIDS orphan Catholic Church and HIV / AIDS Circumcision and HIV Criminal transmission Discrimination against people Economic
impact HIV - affected community HIV / AIDS denialism Safe sex Sex education List of HIV - positive people People With AIDS Self - Empowerment Movement Culture
International AIDS Conference International AIDS Society Joint United Nations Programme on HIV / AIDS ( UNAIDS ) President 's Emergency Plan for AIDS Relief (
PEPFAR ) Treatment Action Campaign World AIDS Day Media portrayal of HIV / AIDS Misconceptions about HIV /
</item 1>

<item 2>
Sub-Saharan Africa
In response , a number of initiatives have been launched to educate the public on HIV / AIDS . Among these are combination prevention programmes , considered to
be the most effective initiative , the abstinence , be faithful , use a condom campaign , and the Desmond Tutu HIV Foundation 's outreach programs . According
to a 2013 special report issued by the Joint United Nations Programme on HIV / AIDS ( UNAIDS ) , the number of HIV positive people in Africa receiving
anti-retroviral treatment in 2012 was over seven times the number receiving treatment in 2005 , with an almost 1 million added in the last year alone . The
number of AIDS - related deaths in Sub-Saharan Africa in 2011 was 33 percent less than the number in 2005 . The number of new HIV infections in Sub-Saharan
Africa in 2011 was 25 percent less than the number in 2001 . Estimated prevalence in % of HIV among young adults ( 15 -- 49 ) per country as of 2011 . Malaria
is an endemic illness in Sub-Saharan Africa , where the majority of malaria cases and deaths worldwide occur . Routine immunization has been introduced in order
to prevent measles .
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
</item 4>

<item 5>
List of countries by HIV/AIDS adult prevalence rate
30 % 11,000 - 2016 Mexico 0.30 % 220,000 4,200 2016 Armenia 0.20 % 3,300 - 2016 Ireland 0.20 % 6,200 - 2016 Kazakhstan 0.20 % 26,000 - 2016 Kyrgyzstan 0.20 %
8,500 - 2016 Lithuania 0.20 % 2,900 - 2016 Madagascar 0.20 % 31,000 1,600 2016 Nepal 0.20 % 32,000 1,700 2016 Netherlands 0.20 % 23,000 - 2016 Nicaragua 0.20 %
8,900 - 2016 Sudan 0.20 % 56,000 3,000 2016 Sweden 0.20 % 11,000 - 2016 Albania 0.10 % 1,700 - 2016 Australia 0.10 % 25,000 - 2016 Fiji 0.10 % - - 2016 Taiwan
0.10 % 31,
</item 5>

<item 6>
List of countries by HIV/AIDS adult prevalence rate
200 - 2016 Rwanda 3.10 % 220,000 3,300 2016 Congo , Republic of the 3.10 % 91,000 3,800 2016 Guinea - Bissau 3.10 % 36,000 2,000 2016 Nigeria 2.90 % 3,200,000
160,000 2016 Cote d'Ivoire 2.70 % 460,000 25,000 2016 South Sudan 2.70 % 200,000 13,000 2016 Haiti 2.10 % 150,000 4,600 2016 Togo 2.10 % 100,000 5,100 2016
Angola 1.90 % 280,000 11,000 2016 Belize 1.80 % 4,300 - 2016 Sierra Leone 1.70 % 67,000 2,800 2016 Gambia , The 1.70 % 20,000 1,100 2016 Jamaica 1.70 % 30,000
1,
</item 6>

<item 7>
List of countries by HIV/AIDS adult prevalence rate
40 % 25,000 - 2016 Costa Rica 0.40 % 13,000 - 2016 Colombia 0.40 % 120,000 2,800 2016 Vietnam 0.40 % 250,000 8,000 2016 Argentina 0.40 % 120,000 2,400 2016
Belarus 0.40 % 19,000 - 2016 USA 0.30 % 1,100,000 - 2015 Somalia 2016 Senegal 0.40 % 41,000 1,900 2016 India 0.30 % 2,100,000 62,000 2016 Ecuador 0.30 % 33,000
- 2016 Bolivia 0.30 % 19,000 - 2016 Tajikistan 0.30 % 14,000 - 2016 Peru 0.30 % 70,000 2,200 2016 Italy 0.30 % 130,000 - 2016 Laos 0.30 % 11,000 - 2016 Mexico
0.30 % 220,
</item 7>

<item 8>
List of countries by HIV/AIDS adult prevalence rate
AVERT . Retrieved 3 August 2017 . Jump up ^ http://factbook.lincon.com/geos/print/country/countrypdf_bf.pdf Jump up ^ `` UNAIDS.org - HIV and AIDS estimates (
2013 ) '' . Jump up ^ `` The Hindu - India has 3rd - highest number of HIV - infected people : UN '' . Jump up ^ `` The Ministry of Health has Detected ... '' .
Bhutan Observer online . 2011 - 08 - 01 . Archived from the original on 2011 - 10 - 28 . Retrieved 2011 - 11 - 21 . Jump up ^ `` CIA world factbook - country
comparison '' . Jump up ^ https://www.cdc.gov/hiv/statistics/overview/index.html Jump up ^ https://www.cdc.gov.tw/downloadfile.aspx?fid=40CE068DC7A0F111
External links ( edit ) HIV / AIDS Survey Indicators Database World Health Organisation ; Prevalence of HIV among adults aged 15 to 49 ( % ) by country World
Health Organisation ;
</item 8>

<item 9>
List of countries by HIV/AIDS adult prevalence rate
) Treatment Action Campaign World AIDS Day Media portrayal of HIV / AIDS Misconceptions about HIV / AIDS Discredited HIV / AIDS origins theories ( show ) AIDS
pandemic by region / country Africa Angola Benin Botswana Democratic Republic of the Congo Egypt Ethiopia Ghana Guinea Côte d'Ivoire ( Ivory Coast ) Kenya
Lesotho Madagascar Malawi Mali Mozambique Namibia Niger Nigeria Rwanda Senegal Swaziland Tanzania South Africa Uganda Zambia Zimbabwe North America Canada
Mexico El Salvador Guatemala Honduras Nicaragua United States New York City Caribbean Haiti Jamaica Dominican Republic South America Bolivia Brazil Colombia
Guyana Peru Asia Bangladesh Bhutan Myanmar ( Burma ) Cambodia China ( PRC ) ( Yunnan ) East Timor India Indonesia Iraq Japan Jordan Laos Nepal North Korea
Pakistan Philippines Thailand Taiwan ( ROC ) United Arab Emirates Vietnam Europe United Kingdom Russia Ukraine Oceania Australia New Zealand Papua New Guinea
List of countries by HIV / AIDS adult prevalence rate List of HIV / AIDS cases and deaths registered by region ( show ) HIV / AIDS in Africa Sovereign states
Algeria Angola Benin Botswana Burkina Faso Burundi Cameroon Cape Verde ( Cabo Verde ) Central African Republic Chad Comoros Democratic Republic of the Congo
Republic of the Congo Djibouti Egypt Equatorial
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
=== END ITEM id=328d55daa0710995 ===

=== ITEM id=e4f7238e742cea40 ===
Gold sentences and facts:
Sentence 0: Third Generation ( GK Series ) Overview Production 2013 -- present Model years 2014 -- present Assembly Yorii , Japan Sumaré , Brazil Ayutthaya , Thailand Pingtung , Taiwan Karawang , Indonesia Malacca , Malaysia Guangzhou , China ( Guangqi Honda ) Celaya , Guanajuato , Mexico ( from Feb 2014 ) Tapukara , India ( from Jan 2015 ) Designer Riku Wada ( 2010 ) Body and chassis Related Honda City Honda HR - V Honda Freed Honda Shuttle Acura CDX Powertrain Engine 1.2 L L12B i - VTEC I4 petrol 1.3 L L13B Earth Dreams i - VTEC I4 1.5 L L15A I4 ( petrol ) 1.5 L L15B1 I4 ( petrol ) 1.5 L L15B Earth Dreams i - VTEC I4 1.5 L i - DTEC diesel Transmission 5 - speed manual ( China Only ) 6 - speed manual 7 - speed semi-automatic ( DCT ) CVT automatic Dimensions Wheelbase 2,530 mm ( 99.6 in ) Length 3,955 mm ( 155.7 in ) Width 1,695 mm ( 66.7 in ) Height 1,525 -- 1,550 mm ( 60.0 -- 61.0 in ) Curb weight 1,020 -- 1,080 kg ( 2,249 -- 2,381 lb ) 1,080 -- 1,170 kg ( 2,381 -- 2,579 lb ) ( Hybrid )
  Fact 0.0: The Third Generation (GK Series) has production from 2013 to the present.
  Fact 0.1: The Third Generation (GK Series) has model years from 2014 to the present.
  Fact 0.2: The Third Generation (GK Series) is assembled in Yorii, Japan.
  Fact 0.3: The Third Generation (GK Series) is assembled in Sumaré, Brazil.
  Fact 0.4: The Third Generation (GK Series) is assembled in Ayutthaya, Thailand.
  Fact 0.5: The Third Generation (GK Series) is assembled in Pingtung, Taiwan.
  Fact 0.6: The Third Generation (GK Series) is assembled in Karawang, Indonesia.
  Fact 0.7: The Third Generation (GK Series) is assembled in Malacca, Malaysia.
  Fact 0.8: The Third Generation (GK Series) is assembled in Guangzhou, China (Guangqi Honda).
  Fact 0.9: The Third Generation (GK Series) is assembled in Celaya, Guanajuato, Mexico, from February 2014.
  Fact 0.10: The Third Generation (GK Series) is assembled in Tapukara, India, from January 2015.
  Fact 0.11: The Third Generation (GK Series) was designed by Riku Wada in 2010.
  Fact 0.12: The Third Generation (GK Series) is related to the Honda City.
  Fact 0.13: The Third Generation (GK Series) is related to the Honda HR-V.
  Fact 0.14: The Third Generation (GK Series) is related to the Honda Freed.
  Fact 0.15: The Third Generation (GK Series) is related to the Honda Shuttle.
  Fact 0.16: The Third Generation (GK Series) is related to the Acura CDX.
  Fact 0.17: The Third Generation (GK Series) has a 1.2 L L12B i-VTEC I4 petrol engine.
  Fact 0.18: The Third Generation (GK Series) has a 1.3 L L13B Earth Dreams i-VTEC I4 engine.
  Fact 0.19: The Third Generation (GK Series) has a 1.5 L L15A I4 petrol engine.
  Fact 0.20: The Third Generation (GK Series) has a 1.5 L L15B1 I4 petrol engine.
  Fact 0.21: The Third Generation (GK Series) has a 1.5 L L15B Earth Dreams i-VTEC I4 engine.
  Fact 0.22: The Third Generation (GK Series) has a 1.5 L i-DTEC diesel engine.
  Fact 0.23: The 5-speed manual transmission is available in China only.
  Fact 0.24: The Third Generation (GK Series) has a 6-speed manual transmission.
  Fact 0.25: The Third Generation (GK Series) has a 7-speed semi-automatic (DCT) transmission.
  Fact 0.26: The Third Generation (GK Series) has a CVT automatic transmission.
  Fact 0.27: The Third Generation (GK Series) has a wheelbase of 2,530 mm (99.6 in).
  Fact 0.28: The Third Generation (GK Series) has a length of 3,955 mm (155.7 in).
  Fact 0.29: The Third Generation (GK Series) has a width of 1,695 mm (66.7 in).
  Fact 0.30: The Third Generation (GK Series) has a height of 1,525 to 1,550 mm (60.0 to 61.0 in).
  Fact 0.31: The Third Generation (GK Series) has a curb weight of 1,020 to 1,080 kg (2,249 to 2,381 lb).
  Fact 0.32: The hybrid version of the Third Generation (GK Series) has a curb weight of 1,080 to 1,170 kg (2,381 to 2,579 lb).

Retrieved items:
<item 1>
Honda Fit
A total of 56,432 Fits were sold in the United States in 2007 . The 2008 Honda Fit was largely unchanged from the 2007 with the exception of the TPMS and new
windshield wipers that abandon the large rubber blade design to a more traditional rigid plastic with thin rubber blade design . MSRP was also increased by
roughly US $100 . Total sales in the United States for the year were 79,794 , an increase of almost 41 % over previous year . Reception Honda marketed the Fit
as `` Small is the New Big '' . Car and Driver magazine described that `` ( n ) ot only is it endowed with an impressive array of standard features and an
upscale interior , but its dynamics rival the responses of cars with much fancier pedigrees and price tags . '' The magazine even cited its lane - change number
of 71.4 mph edged the best run by a Ferrari F430 Spider F1 tested . And the car was included in Car and Driver 's 10 Best Lists for three times . In May 2006 ,
Fit was placed first in a Car and Driver comparison of seven economy cars that included Nissan Versa and Toyota Yaris .
</item 1>

<item 2>
Honda Fit
Indonesia Santa Rosa , Philippines Guangzhou , China ( Guangqi Honda ) Sumaré , Brazil ( Honda Brazil ) Designer Mitsuhiro Honda ( 1999 ) Body and chassis
Related Honda Airwave Honda City / Fit Aria Powertrain Engine 1.2 L L12A I4 ( petrol ) 1.3 L L13A I4 ( petrol ) 1.5 L L15A I4 ( petrol ) Transmission 5 - speed
manual 5 - speed automatic CVT automatic Dimensions Wheelbase 2,450 mm ( 96.5 in ) Length 3,845 mm ( 151.4 in ) ( Worldwide ) 4,000 mm ( 157.4 in ) ( U.S. &
Canada ) Width 1,675 mm ( 65.9 in ) Height 1,525 mm ( 60.0 in ) Curb weight 1,084 kg ( 2,390 lb ) The Honda Fit debuted in June 2001 in Japan and immediately
became a big hit . At its introduction in 2001 , it won the Car of the Year Japan Award .
</item 2>

<item 3>
Honda Fit
National Highway Traffic Safety Administration ( NHTSA ) crash test ratings . The engine is a 1.5 L ( 92 cu in ) i - VTEC 4 - cylinder with 117 horsepower at
6600 rpm mated to a five - speed manual transmission or a five - speed automatic transmission . The second generation Honda Fit went on sale in North America
around August 26 , 2008 , for the 2009 model year . It was considered the most expensive within the subcompact class when equipped as such , but was also
described as modestly priced . It was reported in September 2009 that Honda is considering producing the Fit in North America in response to the increasing
strength of the Japanese yen against the U.S. dollar . For the 2011 models , Vehicle Stability Assist , cruise control , remote entry and USB audio interface
became standard across all trim levels . In addition , the Fit Sport now comes with carpeted floor mats .
</item 3>

<item 4>
The Honda Fit debuted in Japan in June 2001 and won Japan’s Car of the Year award. Its listed production sites include Santa Rosa, Guangzhou and Sumaré; later
sites include Celaya and Tapukara. Designers listed are Mitsuhiro Honda (1999) and Riku Wada (2010). Petrol engines range from 1.2 to 1.5 L, with a later 1.5 L
diesel option; transmissions include manual, automatic
</item 4>

<item 5>
The Honda Fit launched in the U.S. and Canada in April 2006 as a 2007 model, replacing North America’s Civic hatchback. Safety equipment included power windows,
side and curtain airbags, and ABS; North American models had larger bumpers and a longer front end. All trims used a 109-hp 1.5-liter engine with a five-speed
manual or optional five-speed automatic; Sport automatics added paddle shifters. U.S. sales rose from
</item 5>

<item 6>
The references describe the Honda Fit’s safety equipment, pricing, popularity, production, and model-year updates. Reports noted possible safety-feature
reductions tied to rebates, strong dealer demand, and 2008 sales results. The Fit was praised among subcompacts and included in a “10Best Cars” list; a top-trim
2009 model with navigation cost $18,760. Honda considered shifting production to Ohio, added Vehicle Stability Assist as standard on all 2011 models, and
announced
</item 6>

<item 7>
Honda Fit
Gawker Media . Retrieved 2008 - 03 - 19 . Jump up ^ Bell , Damon . `` The Consumer Guide to 2009 Subcompact Cars '' . Consumer Guide Automotive . HowStuffWorks
. Archived from the original on 2009 - 08 - 30 . Retrieved 2009 - 08 - 25 . The most expensive subcompact , the top - trim - level Honda Fit with an optional
navigation system , has a still - modest base price of $18,760 . Jump up ^ `` Honda Fit production could shift to Ohio '' . The Columbus Dispatch . Archived
from the original on 2013 - 01 - 21 . Retrieved 2009 - 10 - 22 . Jump up ^ `` 2011 Honda Fit Receives More Standard Features ; VSA Added to All Models '' .
theautochannel.com . Jump up ^ `` 2009 Honda Fit Review '' . JB car pages . Retrieved 2009 - 03 - 09 . Jump up ^ `` 2012 Honda Fit Receives Updates and More
Features '' ( Press release ) .
</item 7>

<item 8>
Honda Fit
525 -- 1,550 mm ( 60.0 -- 61.0 in ) Curb weight 1,020 -- 1,080 kg ( 2,249 -- 2,381 lb ) 1,080 -- 1,170 kg ( 2,381 -- 2,579 lb ) ( Hybrid ) The third generation
Fit retains the overall design concept of previous generations , notably the center - located fuel tank and multi-configuration interior concept , marketed as
Magic Seating . Honda 's all - new `` Global Small Car Platform '' employing ultra-high strength 780 MPa yield steel for 27 % of its bodywork -- and a shorter
overall length ( - 1.6 inches ) , longer wheelbase ( + 1.2 inch ) increased rear legroom ( + 4.8 inches ) and increased passenger volume ( + 4.9 cu ft )
compared to its previous generation . Body panels are both welded and bolted to the frame in a hybrid monocoque and spaceframe fusion -- and rear torsion beam
suspension is more compact , no longer using an anti-sway bar to maximize interior and cargo space . The HR - V shares its platform with the Fit .
</item 8>

<item 9>
The Honda Fit is categorized as a Honda vehicle introduced in 2001, spanning the 2000s and 2010s. Categories include mini MPVs, subcompact cars, hatchbacks,
hybrids, battery-electric vehicles, and front- or all-wheel-drive cars, including models with CVT transmissions. The Wikipedia page was last edited April 11,
2018, and notes several unsourced statements.
</item 9>

<item 10>
The references cover 2008 and 2009 Honda Fit specifications, an Edmunds first drive of the 2009 Fit, Honda’s 2012 goal to nearly double car sales in four years,
and Bloomberg’s 2010 report on the Fit Hybrid and possible overseas production.
</item 10>
=== END ITEM id=e4f7238e742cea40 ===

=== ITEM id=d2f6521da5f57b8c ===
Gold sentences and facts:
Sentence 0: No . overall No. in series Title Vehicles tested Special guests Original air date 14 `` Past , Present , or Future '' Lamborghini Aventador S , Honda NSX , Rimac Concept One David Hasselhoff , Ricky Wilson 8 December 2017 ( 2017 - 12 - 08 ) The tent is now in the Cotswolds and will remain there for the duration of the series . To determine whether the past , present , or future of motoring is best , the presenters visit Switzerland with Clarkson in a petrol powered Lamborghini Aventador S representing the past , May in a hybrid powered Honda NSX representing the present , and Hammond in an all - electric Rimac Concept One representing the future . After starting at the top of the Gotthard Pass , Hammond brings the trio to a stay at the Park Weggis ( de ) health retreat and visit the Swiss Museum of Transport and other museums in Lucerne close to a charging station for the Rimac car . May and Clarkson reject this and go to a hill climbing race at Hemberg instead . As Clarkson can not fit into the Aventador whilst wearing a mandatory racing helmet , Clarkson sends a Lamborghini test driver in his place , thus setting the fastest time of all three . Hammond crashes after completing the course , destroying the Rimac in a fire . Footage taken from YouTube shows the wreckage being extinguished and Hammond being airlifted by the Swiss Air - Rescue ( REGA ) to hospital . The presenters discuss some inapropriate new features within cars on Conversation Street . Celebrity Brain Crash is replaced by Celebrity Face Off , in which Ricky Wilson beats David Hasselhoff to determine who the `` fastest former talent show host '' is . 15 `` The Falls Guys '' Ford GT , Mercedes AMG GT R Kevin Pietersen , Brian Wilson 15 December 2017 ( 2017 - 12 - 15 ) The presenters race from Central Park in Manhattan , New York City , to an observation tower overlooking the Niagara Falls . Clarkson drives the new Ford GT whilst May and Hammond ( on crutches ) use public transport , taking the New York City Subway , AirTrain JFK , a JetBlue Embraer E190 flight , plus two Niagara Frontier Transportation Authority buses . Despite Hammond and May 's flight arriving 15 minutes early , Clarkson narrowly wins the race . For Conversation Street the presenters discuss sponsorship ideas . Clarkson reviews the Mercedes AMG GT R , after which new test driver , Abbie Eaton , takes it for a timed lap around the Eboladrome . In `` Celebrity Face Off '' , Kevin Pietersen of South Africa beats Brian Wilson of the U.S. as the `` fastest person who makes a living throwing and hitting and catching small balls '' . 16 `` Bah Humbug - atti '' Kia Stinger GT , Bugatti Chiron Hugh Bonneville , Casey Anderson 22 December 2017 ( 2017 - 12 - 22 ) The presenters begin with Christmas tree lights made from vehicle parts and , Starting with Scalextric , they exchange humorous and inappropriate Christmas gifts in Conversation Street . May travels to Majorca to race a Kia Stinger GT against a pair of longboard riders along a mountain road , with May winning the race despite him racing uphill and the riders downhill . Hammond and May attempt to relieve lunchtime boredom by inventing `` lunchtime office car park racing '' at Houghton Hall business park in the Luton / Dunstable Urban Area . After brief interviews in which both share their own ownership of a Tesla Model S , Hugh Bonneville , representing the UK , and Casey Anderson , representing the United States , compete during `` Celebrity Face Off '' as the `` fastest person who lives with a bear '' , with Anderson winning by setting the faster lap time . Clarkson experiments with the jet set lifestyle , starting in the morning from the Da Vinci super-yacht moored in Saint - Tropez and then driving a Bugatti Chiron to L'Alpe De Venosc France for skiing at lunchtime , stopping briefly for a drag race against a BMW M4 at Gap -- Tallard Airport to give viewers a perspective on how fast the Chiron is . After leaving the Alps , Clarkson then pays an homage to the 1976 film C'était un rendez - vous , by speeding across Turin and reaching the Piazza Carignano ( it ) in 9 minutes . 17 `` Unscripted '' McLaren 720S , Audi TT RS , Ariel Nomad Michael Ball , Alfie Boe , Silvija Jurin 29 December 2017 ( 2017 - 12 - 29 ) The presenters begin by having a year - ending awards ceremony in Conversation Street . Hammond heads to the Eboladrome to test the McLaren 720S after which Eaton sets a lap time . Since The Grand Tour has been criticised for being too `` scripted '' , the presenters head to Zadar county and the island of Pag in Croatia to make a completely `` unscripted '' film . This results in Clarkson turning up in the Audi TT RS , Hammond turning up in the not at all comparable Ariel Nomad , and May turning up in an old Lada Riva which he turns into a fire engine . They go their separate ways though Clarkson and Hammond end up comparing their cars in a drag race , a rally stage ( both of which are won by Clarkson ) , visiting Maslenica , the Plitvice Lakes National Park , and following the D106 road until running out of road at Žigljen ferry port . Due to a lack of fires , May has to setup his own ones until Clarkson and Hammond setup one that proves too difficult for May , which the real fire brigade manage to put out with a firefighting aircraft , destroying May 's Lada fire engine in the process . Michael Ball and Alfie Boe compete in `` Celebrity Face Off '' with Ball setting the faster time and becoming the `` fastest classical singer with some kind of interesting connection to the British motor industry '' . 18 5 `` Up , Down and Round the Farm '' Ripsaw EV2 , Volkswagen Up ! GTI Bill Bailey , Dominic Cooper , Mark Higgins 5 January 2018 ( 2018 - 01 - 05 ) The presenters introduce some of the new cars for 2018 in Conversation Street . Hammond heads to Dubai to test a Sandrail dune buggy in the desert , followed by the Ripsaw `` high - end luxury tank '' , visiting the Dubai Mall . May reviews the Volkswagen Up ! GTI at the Eboladrome and in thick fog Eaton records a record slow time . Guests Bill Bailey and Dominic Cooper discuss their shared ownership of old French cars before racing each other in greasy / wet conditions with Cooper becoming the `` fastest person who has a failed former career in a band '' . Clarkson creates a Ken Block - style `` Farmkhana '' film using an old Subaru Impreza WRX STi around Hare Farm , near Rye , with bullet time photography . Hammond and May show the audience that Clarkson 's driving was mostly fake by sharing a `` making - of '' consisting of outtakes and behind - the - scenes footage featuring rally driver Mark Higgins , producer Phil Churchward , plus camera operator Ben Joiner breaking the fourth wall . Clarkson suggests that the whole thing is like sawing a woman in half during a magic show , and that really the audience do n't want to know . 19 6 `` Jaaaaaaaags '' None Luke Evans , Kiefer Sutherland 12 January 2018 ( 2018 - 01 - 12 ) The presenters head to the area around Grand Junction , Colorado , staying at the Gateway Canyons Resort & Spa , and eventually travelling to Telluride , Colorado with a variety of old Jaguar cars to establish the manufacturer 's reputation for reliability with Clarkson buying an XJR , May taking an XK8 convertible and Hammond bringing a Mark X . During the course of the trip the cars are driven around a dirt track ( where Hammond 's car is badly damaged , and is replaced by an XJ6 ) , along the runway at Telluride Regional Airport , ( where May 's car is destroyed , and is replaced by an XJS ) and down the slopes of the Telluride Ski Resort . At Hanging Flume overlook , Clarkson experiences a traffic stop by a Mesa County police officer and reads the `` Her Britannic Majesty '' text from his British passport . Despite Clarkson 's car being the best , May admitting his car had poor brakes deemed the challenge pointless . The presenters discuss the annoyance of Firemen cutting the roofs of cars in Conversation Street . Kiefer Sutherland beats Luke Evans to become `` the fastest actor with a middle name of George '' . 20 7 `` It 's a Gas , Gas , Gas '' Lamborghini Huracán Performante Anthony Joshua , Bill Goldberg 19 January 2018 ( 2018 - 01 - 19 ) The presenters discuss some new supercars and how handbrake turns impress women within the Conversation Street Segment . Hammond tests the Lamborghini Huracán Performante at the Eboladrome beating May 's Ferrari 458 Speciale in a drag race before Eaton takes it round the track setting a fast time . May and Hammond then look at ingenious ways of filling up with fuel instead of stopping at a fuel station . Anthony Joshua becomes `` the fastest man who makes a living out of punching or strangling someone '' beating Bill Goldberg , with both previously owning cars from General Motors . Clarkson then talks about the Rivalry in rallying between Audi with the Quattro and Lancia with the 037 . They conclude with a `` terrible disappointment '' that Lancia is no longer around . 21 8 `` Blasts from the Past '' Jaguar XK SS , Aston Martin DB4 GT , Honda Civic Type R , Ford GT Stewart Copeland , Nick Mason 26 January 2018 ( 2018 - 01 - 26 ) The presenters head to the City of Pau , Pyrénées - Atlantiques , France , and home of the Pau Grand Prix on the Circuit de Pau - Ville ( fr ) street course . At Palais Beaumont ( fr ) Hammond arrives with a Jagaur XK SS and Clarkson in an Aston Martin DB4 GT ; both being modern builds of older designs from the 1950s `` back - catalogues '' of Jaguar and Aston Martin . They both race around the Streets of Pau before May turns up in a modern Honda Civic Type R to prove that new cars are better . All three depart to drive south over the Pyrenees mountain range , with Clarkson and Hammond experiencing difficulties in dense fog . Clarkson recants a previous incident from 2013 -- when Clarkson and Hammond 's driving licences were confiscated whilst filming The Perfect Road Trip in France -- and tries to goad May into speeding on the Autopista AP - 2 . At Barcelona the presenters race around the abandoned Autódromo de Sitges - Terramar and the extreme banking on its corners , with May winning , and Hammond abstaining following his history of crashes . During Conversation Street the presenters discuss names for gritters . At the Eboladrome , Clarkson retests the Ford GT following the Niagara Falls race , with Eaton setting a test lap . Nick Mason goes on to record a faster lap than Stewart Copeland to become `` the fastest drummer for a band beginning with the letter p '' . 22 9 `` Breaking , Badly '' Jaguar XJ220 , Bugatti EB 110 Super Sport Dynamo , Penn & Teller 2 February 2018 ( 2018 - 02 - 02 ) The presenters discuss new sport cars by various companies , jumpers and the speed of pigeons vs blackbirds during Conversation Street . Clarkson unsuccessfully sets out to prove that mankind has not significantly improved over the last 25 years by testing the Jaguar XJ220 and the Bugatti EB 110 Super Sport , the fastest supercars from the early 90s , both of which set slow times at the Eboladrome when driven by Eaton . The trio tries to break the UK speed record for amphibious vehicles by building one of their own . Clarkson oversees his idea of combining a Rolls - Royce Spey turbofan engine from a Nimrod jet and a speedboat with a Suzuki Jimny . However on a nearby lake , Clarkson and May have to abort their run after achieving only 3 mph ( 5 km / h ) and nearly sinking . Hammond and May then go on to build a vehicle of their own by combining a Bond Bug with the engine of a personal water craft nicknaming it the `` Pond Bug '' . Competing on Coniston Water during Coniston Power Boat Records Week , Clarkson , being the only one with a speedboat licence , operates the `` Pond Bug '' and achieves an average of 47.81 mph ( 76.94 km / h ) , setting a new UK record . After performing a series of magic tricks , Dynamo and Penn & Teller compete to become `` the fastest magician in the world '' , with Penn & Teller prevailing on a snowy track . 23 10 `` Oh , Canada '' Alfa Romeo Stelvio Quadrifoglio , Range Rover Velar P380 , Porsche Macan Turbo Performance Pack , Tesla Model X , Ford F150 Raptor Rory McIlroy , Paris Hilton 9 February 2018 ( 2018 - 02 - 09 ) The presenters head to Canada to prove that small SUV 's are useless with Clarkson picking an Alfa Romeo Stelvio Quadrifoglio , May choosing a Range Rover Velar P380 and Hammond taking a Porsche Macan Turbo Performance Pack . They are shown to be slower round the Area 27 circuit than a BMW M3 , not practical enough to carry Canadian Newfoundland dogs , and slower round a barrel course than a horse : concluding that SUVs are pointless . Despite this , executive producer Andy Wilman informs via text that they will race their SUVs whilst each towing a boat up a mountain . May abandons his Velar for his boat , which sinks , causing him to lose the race . Because Clarkson 's Stelvio has no tow bar he cheats and exchanges it for a Ford F150 Raptor , ultimately arriving first with the presenters concluding that the Raptor was better than all the SUVs being tested . On Conversation Street the presenters talk about new SUVs and vans . Clarkson reviews the Tesla Model X at the Eboladrome and on the road , accompanied by some lawyers , ostensibly to avoid Tesla pursuing any legal action , following the previous Tesla v. Top Gear court cases . Rory McIlroy and Paris Hilton discuss their car collections before McIlroy beats Hilton to become the `` World 's Fastest Golfing Enthusiast '' . 24 11 `` Feed the World '' None None 16 February 2018 ( 2018 - 02 - 16 ) The presenters head to Mozambique on a mission to feed the people of Bingo , an isolated town in the west of the country , with fish from the country 's capital of Maputo , covering a distance of approximately 200 miles . To do this , each of them attempt to find the best transport solution : May buys a Mercedes - Benz Estate and modifies it by fitting a Perspex tank filled with sea water in which to keep the fish alive ; Clarkson buys a Nissan Hardbody truck which he fits with a generator and an ice machine in an attempt to keep the fish frozen and Hammond buys a TVS Star motorcycle and devises a rack which he fits to the bike 's rear , inspired by a local method of drying and salting the fish to preserve them . After unsuccessfully trying to catch fish using local methods , they buy fish at a local market . Early in the journey , Clarkson 's truck crashes into May 's car upon rescuing it , damaging its radiator grille . Deeper into rural Mozambique , May constantly soaks himself with sea water from his ill - fitted tank and , due to his damaged radiator grille , suffers multiple breakdowns due to water intrusion . Hammond loses a large number of his fish on the bumpy roads , constantly gets stuck in thick mud and endlessly falls from his motorbike . Clarkson suffers very few issues except when his ice machine breaks , causing him to make small adjustments to his truck in order to smoke the fish with his exhaust . Upon May 's car breaking down in the middle of a lake , despite Hammond and Clarkson crossing successfully , Clarkson cuts off the front end of the Mercedes and tows what remains of the rear portion using his truck - much to May 's dismay and annoyance . Eventually , the three reach Bingo with May 's fish dead , Clarkson 's fish ruined and Hammond having almost no fish left . As a result , they sell no fish and leave Bingo in a helicopter `` like all charitable celebrities '' .
  Fact 0.0: Episode 14 is titled Past, Present, or Future.
  Fact 0.1: Episode 14 is numbered 14 overall and 14 in the series.
  Fact 0.2: Episode 14 tests the Lamborghini Aventador S, Honda NSX, and Rimac Concept One.
  Fact 0.3: Episode 14 has special guests David Hasselhoff and Ricky Wilson.
  Fact 0.4: Episode 14 originally aired on 8 December 2017.
  Fact 0.5: The tent is now in the Cotswolds and will remain there for the duration of the series.
  Fact 0.6: The presenters want to determine whether the past, present, or future of motoring is best.
  Fact 0.7: The presenters visit Switzerland with Clarkson in a petrol powered Lamborghini Aventador S representing the past.
  Fact 0.8: May drives a hybrid powered Honda NSX representing the present.
  Fact 0.9: Hammond drives an all-electric Rimac Concept One representing the future.
  Fact 0.10: After starting at the top of the Gotthard Pass, Hammond brings the trio to a stay at the Park Weggis (de) health retreat.
  Fact 0.11: Hammond visits the Swiss Museum of Transport and other museums in Lucerne.
  Fact 0.12: The Park Weggis stay is close to a charging station for the Rimac car.
  Fact 0.13: May and Clarkson reject the plan and go to a hill climbing race at Hemberg instead.
  Fact 0.14: Clarkson cannot fit into the Aventador whilst wearing a mandatory racing helmet.
  Fact 0.15: Clarkson sends a Lamborghini test driver in his place.
  Fact 0.16: The Lamborghini test driver sets the fastest time of all three.
  Fact 0.17: Hammond crashes after completing the course, destroying the Rimac in a fire.
  Fact 0.18: Footage taken from YouTube shows the wreckage being extinguished.
  Fact 0.19: Footage taken from YouTube shows Hammond being airlifted by the Swiss Air-Rescue (REGA) to hospital.
  Fact 0.20: The presenters discuss some inappropriate new features within cars on Conversation Street.
  Fact 0.21: Celebrity Brain Crash is replaced by Celebrity Face Off.
  Fact 0.22: In Celebrity Face Off, Ricky Wilson beats David Hasselhoff to determine who the fastest former talent show host is.
  Fact 0.23: Episode 15 is titled The Falls Guys.
  Fact 0.24: Episode 15 is numbered 15 overall and 15 in the series.
  Fact 0.25: Episode 15 tests the Ford GT and Mercedes AMG GT R.
  Fact 0.26: Episode 15 has special guests Kevin Pietersen and Brian Wilson.
  Fact 0.27: Episode 15 originally aired on 15 December 2017.
  Fact 0.28: The presenters race from Central Park in Manhattan, New York City, to an observation tower overlooking the Niagara Falls.
  Fact 0.29: Clarkson drives the new Ford GT.
  Fact 0.30: May and Hammond use public transport, with Hammond on crutches.
  Fact 0.31: May and Hammond take the New York City Subway, AirTrain JFK, a JetBlue Embraer E190 flight, and two Niagara Frontier Transportation Authority buses.
  Fact 0.32: Hammond and May's flight arrives 15 minutes early.
  Fact 0.33: Clarkson narrowly wins the race.
  Fact 0.34: For Conversation Street the presenters discuss sponsorship ideas.
  Fact 0.35: Clarkson reviews the Mercedes AMG GT R.
  Fact 0.36: After the review, new test driver Abbie Eaton takes the Mercedes AMG GT R for a timed lap around the Eboladrome.
  Fact 0.37: In Celebrity Face Off, Kevin Pietersen of South Africa beats Brian Wilson of the U.S. as the fastest person who makes a living throwing and hitting and catching small balls.
  Fact 0.38: Episode 16 is titled Bah Humbug - atti.
  Fact 0.39: Episode 16 is numbered 16 overall and 16 in the series.
  Fact 0.40: Episode 16 tests the Kia Stinger GT and Bugatti Chiron.
  Fact 0.41: Episode 16 has special guests Hugh Bonneville and Casey Anderson.
  Fact 0.42: Episode 16 originally aired on 22 December 2017.
  Fact 0.43: The presenters begin with Christmas tree lights made from vehicle parts.
  Fact 0.44: Starting with Scalextric, the presenters exchange humorous and inappropriate Christmas gifts in Conversation Street.
  Fact 0.45: May travels to Majorca to race a Kia Stinger GT against a pair of longboard riders along a mountain road.
  Fact 0.46: May wins the race despite him racing uphill and the riders downhill.
  Fact 0.47: Hammond and May attempt to relieve lunchtime boredom by inventing lunchtime office car park racing at Houghton Hall business park in the Luton / Dunstable Urban Area.
  Fact 0.48: After brief interviews, Hugh Bonneville and Casey Anderson each share their own ownership of a Tesla Model S.
  Fact 0.49: Hugh Bonneville represents the UK, and Casey Anderson represents the United States.
  Fact 0.50: Hugh Bonneville and Casey Anderson compete during Celebrity Face Off as the fastest person who lives with a bear.
  Fact 0.51: Casey Anderson wins by setting the faster lap time.
  Fact 0.52: Clarkson experiments with the jet set lifestyle.
  Fact 0.53: Clarkson starts the morning from the Da Vinci super-yacht moored in Saint-Tropez.
  Fact 0.54: Clarkson then drives a Bugatti Chiron to L'Alpe De Venosc France for skiing at lunchtime.
  Fact 0.55: Clarkson stops briefly for a drag race against a BMW M4 at Gap -- Tallard Airport to give viewers a perspective on how fast the Chiron is.
  Fact 0.56: After leaving the Alps, Clarkson pays an homage to the 1976 film C'etait un rendez-vous.
  Fact 0.57: Clarkson speeds across Turin and reaches the Piazza Carignano (it) in 9 minutes.
  Fact 0.58: Episode 17 is titled Unscripted.
  Fact 0.59: Episode 17 is numbered 17 overall and 17 in the series.
  Fact 0.60: Episode 17 tests the McLaren 720S, Audi TT RS, and Ariel Nomad.
  Fact 0.61: Episode 17 has special guests Michael Ball, Alfie Boe, and Silvija Jurin.
  Fact 0.62: Episode 17 originally aired on 29 December 2017.
  Fact 0.63: The presenters begin by having a year-ending awards ceremony in Conversation Street.
  Fact 0.64: Hammond heads to the Eboladrome to test the McLaren 720S.
  Fact 0.65: Eaton sets a lap time after Hammond tests the McLaren 720S.
  Fact 0.66: The Grand Tour has been criticised for being too scripted.
  Fact 0.67: The presenters head to Zadar county and the island of Pag in Croatia to make a completely unscripted film.
  Fact 0.68: Clarkson turns up in the Audi TT RS.
  Fact 0.69: Hammond turns up in the not at all comparable Ariel Nomad.
  Fact 0.70: May turns up in an old Lada Riva which he turns into a fire engine.
  Fact 0.71: The presenters go their separate ways.
  Fact 0.72: Clarkson and Hammond end up comparing their cars in a drag race and a rally stage, both of which are won by Clarkson.
  Fact 0.73: Clarkson and Hammond visit Maslenica, the Plitvice Lakes National Park, and follow the D106 road until running out of road at Zigljen ferry port.
  Fact 0.74: Due to a lack of fires, May has to set up his own ones.
  Fact 0.75: Clarkson and Hammond set up a fire that proves too difficult for May.
  Fact 0.76: The real fire brigade puts out the fire with a firefighting aircraft.
  Fact 0.77: The firefighting aircraft destroys May's Lada fire engine in the process.
  Fact 0.78: In Celebrity Face Off, Michael Ball sets the faster time against Alfie Boe.
  Fact 0.79: Michael Ball becomes the fastest classical singer with some kind of interesting connection to the British motor industry.
  Fact 0.80: Episode 18 is numbered 5 in the series and is titled Up, Down and Round the Farm.
  Fact 0.81: Episode 18 tests the Ripsaw EV2 and Volkswagen Up! GTI.
  Fact 0.82: Episode 18 has special guests Bill Bailey, Dominic Cooper, and Mark Higgins.
  Fact 0.83: Episode 18 originally aired on 5 January 2018.
  Fact 0.84: The presenters introduce some of the new cars for 2018 in Conversation Street.
  Fact 0.85: Hammond heads to Dubai to test a Sandrail dune buggy in the desert.
  Fact 0.86: Hammond then tests the Ripsaw high-end luxury tank.
  Fact 0.87: The presenters visit the Dubai Mall.
  Fact 0.88: May reviews the Volkswagen Up! GTI at the Eboladrome.
  Fact 0.89: In thick fog, Eaton records a record slow time.
  Fact 0.90: Guests Bill Bailey and Dominic Cooper discuss their shared ownership of old French cars.
  Fact 0.91: Bill Bailey and Dominic Cooper race each other in greasy and wet conditions.
  Fact 0.92: Dominic Cooper becomes the fastest person who has a failed former career in a band.
  Fact 0.93: Clarkson creates a Ken Block-style Farmkhana film using an old Subaru Impreza WRX STi around Hare Farm, near Rye.
  Fact 0.94: The Farmkhana film uses bullet time photography.
  Fact 0.95: Hammond and May show the audience that Clarkson's driving was mostly fake.
  Fact 0.96: They share a making-of consisting of outtakes and behind-the-scenes footage.
  Fact 0.97: The making-of features rally driver Mark Higgins, producer Phil Churchward, and camera operator Ben Joiner breaking the fourth wall.
  Fact 0.98: Clarkson suggests that the whole thing is like sawing a woman in half during a magic show.
  Fact 0.99: Clarkson suggests the audience do not really want to know.
  Fact 0.100: Episode 19 is numbered 6 in the series and is titled Jaaaaaaaags.
  Fact 0.101: Episode 19 has no vehicles tested.
  Fact 0.102: Episode 19 has special guests Luke Evans and Kiefer Sutherland.
  Fact 0.103: Episode 19 originally aired on 12 January 2018.
  Fact 0.104: The presenters head to the area around Grand Junction, Colorado, staying at the Gateway Canyons Resort & Spa.
  Fact 0.105: The presenters travel to Telluride, Colorado with a variety of old Jaguar cars.
  Fact 0.106: The trip establishes the manufacturer's reputation for reliability.
  Fact 0.107: Clarkson buys an XJR, May takes an XK8 convertible, and Hammond brings a Mark X.
  Fact 0.108: The cars are driven around a dirt track, where Hammond's car is badly damaged and is replaced by an XJ6.
  Fact 0.109: The cars are driven along the runway at Telluride Regional Airport, where May's car is destroyed and is replaced by an XJS.
  Fact 0.110: The cars are driven down the slopes of the Telluride Ski Resort.
  Fact 0.111: At Hanging Flume overlook, Clarkson experiences a traffic stop by a Mesa County police officer.
  Fact 0.112: Clarkson reads the Her Britannic Majesty text from his British passport.
  Fact 0.113: Despite Clarkson's car being the best, May admits his car had poor brakes.
  Fact 0.114: The presenters deem the challenge pointless.
  Fact 0.115: The presenters discuss the annoyance of Firemen cutting the roofs of cars in Conversation Street.
  Fact 0.116: Kiefer Sutherland beats Luke Evans to become the fastest actor with a middle name of George.
  Fact 0.117: Episode 20 is numbered 7 in the series and is titled It's a Gas, Gas, Gas.
  Fact 0.118: Episode 20 tests the Lamborghini Huracan Performante.
  Fact 0.119: Episode 20 has special guests Anthony Joshua and Bill Goldberg.
  Fact 0.120: Episode 20 originally aired on 19 January 2018.
  Fact 0.121: The presenters discuss new supercars and how handbrake turns impress women within the Conversation Street segment.
  Fact 0.122: Hammond tests the Lamborghini Huracan Performante at the Eboladrome.
  Fact 0.123: Hammond beats May's Ferrari 458 Speciale in a drag race.
  Fact 0.124: Eaton takes the Lamborghini Huracan Performante round the track and sets a fast time.
  Fact 0.125: May and Hammond look at ingenious ways of filling up with fuel instead of stopping at a fuel station.
  Fact 0.126: Anthony Joshua becomes the fastest man who makes a living out of punching or strangling someone.
  Fact 0.127: Anthony Joshua beats Bill Goldberg.
  Fact 0.128: Anthony Joshua and Bill Goldberg both previously owned cars from General Motors.
  Fact 0.129: Clarkson talks about the rivalry in rallying between Audi with the Quattro and Lancia with the 037.
  Fact 0.130: The presenters conclude with a terrible disappointment that Lancia is no longer around.
  Fact 0.131: Episode 21 is numbered 8 in the series and is titled Blasts from the Past.
  Fact 0.132: Episode 21 tests the Jaguar XK SS, Aston Martin DB4 GT, Honda Civic Type R, and Ford GT.
  Fact 0.133: Episode 21 has special guests Stewart Copeland and Nick Mason.
  Fact 0.134: Episode 21 originally aired on 26 January 2018.
  Fact 0.135: The presenters head to the City of Pau, Pyrenees-Atlantiques, France, home of the Pau Grand Prix on the Circuit de Pau-Ville (fr) street course.
  Fact 0.136: At Palais Beaumont (fr), Hammond arrives with a Jaguar XK SS.
  Fact 0.137: Clarkson arrives at Palais Beaumont (fr) in an Aston Martin DB4 GT.
  Fact 0.138: Both cars are modern builds of older designs from the 1950s back-catalogues of Jaguar and Aston Martin.
  Fact 0.139: Hammond and Clarkson race around the Streets of Pau.
  Fact 0.140: May turns up in a modern Honda Civic Type R to prove that new cars are better.
  Fact 0.141: All three drive south over the Pyrenees mountain range.
  Fact 0.142: Clarkson and Hammond experience difficulties in dense fog.
  Fact 0.143: Clarkson recounts a 2013 incident in which Clarkson and Hammond's driving licences were confiscated whilst filming The Perfect Road Trip in France.
  Fact 0.144: Clarkson tries to goad May into speeding on the Autopista AP-2.
  Fact 0.145: At Barcelona, the presenters race around the abandoned Autodromo de Sitges-Terramar and its extreme banking on its corners.
  Fact 0.146: May wins the race at Autodromo de Sitges-Terramar.
  Fact 0.147: Hammond abstains from the race following his history of crashes.
  Fact 0.148: During Conversation Street the presenters discuss names for gritters.
  Fact 0.149: At the Eboladrome, Clarkson retests the Ford GT following the Niagara Falls race.
  Fact 0.150: Eaton sets a test lap.
  Fact 0.151: Nick Mason records a faster lap than Stewart Copeland.
  Fact 0.152: Nick Mason becomes the fastest drummer for a band beginning with the letter P.
  Fact 0.153: Episode 22 is numbered 9 in the series and is titled Breaking, Badly.
  Fact 0.154: Episode 22 tests the Jaguar XJ220 and Bugatti EB 110 Super Sport.
  Fact 0.155: Episode 22 has special guests Dynamo and Penn & Teller.
  Fact 0.156: Episode 22 originally aired on 2 February 2018.
  Fact 0.157: The presenters discuss new sport cars by various companies, jumpers, and the speed of pigeons vs blackbirds during Conversation Street.
  Fact 0.158: Clarkson unsuccessfully sets out to prove that mankind has not significantly improved over the last 25 years.
  Fact 0.159: The Jaguar XJ220 and Bugatti EB 110 Super Sport are the fastest supercars from the early 90s.
  Fact 0.160: Both the Jaguar XJ220 and Bugatti EB 110 Super Sport set slow times at the Eboladrome when driven by Eaton.
  Fact 0.161: The trio tries to break the UK speed record for amphibious vehicles by building one of their own.
  Fact 0.162: Clarkson oversees his idea of combining a Rolls-Royce Spey turbofan engine from a Nimrod jet and a speedboat with a Suzuki Jimny.
  Fact 0.163: On a nearby lake, Clarkson and May have to abort their run after achieving only 3 mph (5 km/h) and nearly sinking.
  Fact 0.164: Hammond and May build a vehicle by combining a Bond Bug with the engine of a personal water craft.
  Fact 0.165: They nickname the vehicle the Pond Bug.
  Fact 0.166: Competing on Coniston Water during Coniston Power Boat Records Week, Clarkson operates the Pond Bug.
  Fact 0.167: Clarkson is the only one with a speedboat licence.
  Fact 0.168: Clarkson achieves an average of 47.81 mph (76.94 km/h), setting a new UK record.
  Fact 0.169: After performing a series of magic tricks, Dynamo and Penn & Teller compete to become the fastest magician in the world.
  Fact 0.170: Penn & Teller prevail on a snowy track.
  Fact 0.171: Episode 23 is numbered 10 in the series and is titled Oh, Canada.
  Fact 0.172: Episode 23 tests the Alfa Romeo Stelvio Quadrifoglio, Range Rover Velar P380, Porsche Macan Turbo Performance Pack, Tesla Model X, and Ford F150 Raptor.
  Fact 0.173: Episode 23 has special guests Rory McIlroy and Paris Hilton.
  Fact 0.174: Episode 23 originally aired on 9 February 2018.
  Fact 0.175: The presenters head to Canada to prove that small SUVs are useless.
  Fact 0.176: Clarkson picks an Alfa Romeo Stelvio Quadrifoglio.
  Fact 0.177: May chooses a Range Rover Velar P380.
  Fact 0.178: Hammond takes a Porsche Macan Turbo Performance Pack.
  Fact 0.179: The SUVs are shown to be slower round the Area 27 circuit than a BMW M3.
  Fact 0.180: The SUVs are not practical enough to carry Canadian Newfoundland dogs.
  Fact 0.181: The SUVs are slower round a barrel course than a horse.
  Fact 0.182: The presenters conclude that SUVs are pointless.
  Fact 0.183: Executive producer Andy Wilman informs via text that they will race their SUVs whilst each towing a boat up a mountain.
  Fact 0.184: May abandons his Velar for his boat, which sinks, causing him to lose the race.
  Fact 0.185: Clarkson's Stelvio has no tow bar, so he cheats and exchanges it for a Ford F150 Raptor.
  Fact 0.186: Clarkson arrives first in the Ford F150 Raptor.
  Fact 0.187: The presenters conclude that the Raptor was better than all the SUVs being tested.
  Fact 0.188: On Conversation Street the presenters talk about new SUVs and vans.
  Fact 0.189: Clarkson reviews the Tesla Model X at the Eboladrome and on the road.
  Fact 0.190: Clarkson is accompanied by some lawyers to avoid Tesla pursuing any legal action.
  Fact 0.191: The lawyers follow the previous Tesla v. Top Gear court cases.
  Fact 0.192: Rory McIlroy and Paris Hilton discuss their car collections.
  Fact 0.193: McIlroy beats Hilton to become the World's Fastest Golfing Enthusiast.
  Fact 0.194: Episode 24 is numbered 11 in the series and is titled Feed the World.
  Fact 0.195: Episode 24 has no vehicles tested and no special guests.
  Fact 0.196: Episode 24 originally aired on 16 February 2018.
  Fact 0.197: The presenters head to Mozambique on a mission to feed the people of Bingo, an isolated town in the west of the country.
  Fact 0.198: The fish come from the country's capital of Maputo.
  Fact 0.199: The fish are carried a distance of approximately 200 miles.
  Fact 0.200: May buys a Mercedes-Benz Estate and fits a Perspex tank filled with sea water in which to keep the fish alive.
  Fact 0.201: Clarkson buys a Nissan Hardbody truck which he fits with a generator and an ice machine to keep the fish frozen.
  Fact 0.202: Hammond buys a TVS Star motorcycle and devises a rack which he fits to the bike's rear.
  Fact 0.203: The rack is inspired by a local method of drying and salting the fish to preserve them.
  Fact 0.204: After unsuccessfully trying to catch fish using local methods, they buy fish at a local market.
  Fact 0.205: Early in the journey, Clarkson's truck crashes into May's car upon rescuing it, damaging its radiator grille.
  Fact 0.206: Deeper into rural Mozambique, May constantly soaks himself with sea water from his ill-fitted tank.
  Fact 0.207: Due to his damaged radiator grille, May suffers multiple breakdowns due to water intrusion.
  Fact 0.208: Hammond loses a large number of his fish on the bumpy roads.
  Fact 0.209: Hammond constantly gets stuck in thick mud and endlessly falls from his motorbike.
  Fact 0.210: Clarkson suffers very few issues except when his ice machine breaks.
  Fact 0.211: Clarkson makes small adjustments to his truck in order to smoke the fish with his exhaust.
  Fact 0.212: When May's car breaks down in the middle of a lake, Hammond and Clarkson cross successfully.
  Fact 0.213: Clarkson cuts off the front end of the Mercedes and tows what remains of the rear portion using his truck, much to May's dismay and annoyance.
  Fact 0.214: The three reach Bingo with May's fish dead.
  Fact 0.215: Clarkson's fish are ruined.
  Fact 0.216: Hammond has almost no fish left.
  Fact 0.217: As a result, they sell no fish.
  Fact 0.218: They leave Bingo in a helicopter like all charitable celebrities.

Retrieved items:
<item 1>
The Grand Tour (TV series)
... we end the day at the Verkehrshaus , aka the Swiss Transport Museum , ... town of Kriens , 20 minutes from Lucerne , ... home to EVTech , ... We follow
Hammond as he plugs in his Rimac and then ... Waits ... why we end the day in the Kriens Chess Museum Jump up ^ Brown , Aaron ( 9 December 2017 ) . `` The Grand
Tour Season 2 : How to watch episode 1 , plus stream FULL second season for Free '' . Sunday Express . Retrieved 24 December 2017 . Dubbed `` Past , Present or
Future '' , this is the episode that features Hammond 's explosive car crash Jump up ^ The Grand Tour ( @ thegrandtour ) ( 5 December 2017 ) . `` 3 days to go
until # TheGrandTour returns to Prime Video with the episode Past , Present , or Future . www.PrimeVideo.com/GT '' ( Tweet ) -- via Twitter . Jump up ^ `` The
Grand Tour Season 2 '' . Amazon.co.uk website . ASIN B0731ZQ3R7. 1 .
</item 1>

<item 2>
The Grand Tour (TV series)
The presenters discuss the annoyance of Firemen cutting the roofs of cars in Conversation Street . Kiefer Sutherland beats Luke Evans to become `` the fastest
actor with a middle name of George '' . 20 7 `` It 's a Gas , Gas , Gas '' Lamborghini Huracán Performante Anthony Joshua , Bill Goldberg 19 January 2018 ( 2018
- 01 - 19 ) The presenters discuss some new supercars and how handbrake turns impress women within the Conversation Street Segment . Hammond tests the
Lamborghini Huracán Performante at the Eboladrome beating May 's Ferrari 458 Speciale in a drag race before Eaton takes it round the track setting a fast time .
May and Hammond then look at ingenious ways of filling up with fuel instead of stopping at a fuel station . Anthony Joshua becomes `` the fastest man who makes
a living out of punching or strangling someone '' beating Bill Goldberg , with both previously owning cars from General Motors . Clarkson then talks about the
Rivalry in rallying between Audi with the Quattro and Lancia with the 037 .
</item 2>

<item 3>
The Grand Tour (TV series)
Rotten Tomatoes . Retrieved 17 July 2017 . Jump up ^ `` The Grand Tour season 2 episode 1 review : We 're in safe hands as Clarkson , Hammond and May return ''
. Digital Spy . Retrieved 8 December 2017 . Jump up ^ `` Television review : The Grand Tour on Amazon Prime '' . The Times . Retrieved 8 December 2017 . Jump up
^ `` The Grand Tour season 2 , episode 1 review : Has Jeremy Clarkson finally found his groove ? '' . The Daily Telegraph . Retrieved 8 December 2017 . Jump up
^ Clarkson , Jeremy ( 8 December 2017 ) . `` Jeremy Clarkson on Twitter '' . twitter.com . Retrieved 15 December 2017 . Jump up ^ Ling , Thomas ( 27 November
2017 ) . `` The Grand Tour 's first celebrity guests for series 2 have been revealed -- and there 's not a single woman '' . Retrieved 20 December 2017 . Jump
up ^ Long , Trevor .
</item 3>

<item 4>
The Grand Tour (TV series)
The Daily Telegraph , while not as positive , still approved of the episode , stating `` The writing is still rather ropey . Clarkson 's suggestion of a new
nickname for May -- `` Dingleberry Handpump '' -- failed to raise a titter even among the super-fans gathered for the London premiere '' but also said that ``
for each wobble , there are just as many moments when The Grand Tour manages the clever trick Top Gear could pull off at its best : raising a chuckle while
sneaking in a bit of serious journalism at the same time . '' and ultimately gave the episode 3 out of 5 stars . Jeremy Clarkson himself believed that they had
`` hit the ground running with series 2 of the Grand Tour '' . Radio Times has criticised the gender balance of guests on the `` Celebrity Face Off '' segment .
There was only 1 woman guest in the episode `` Oh Canada '' ; Paris Hilton . Hothouse Flowers playing for the opening sequence of `` The Holy Trinity '' episode
in the Lucerne Valley , California Broadcast ( edit ) Australian free - to - air network Seven Network started broadcasting the first series of The Grand Tour
in mid-October 2017 .
</item 4>

<item 5>
The Grand Tour (TV series)
The Sun . Retrieved 23 December 2017 . ^ Jump up to : Daly , Emma ( 26 November 2016 ) . `` Jeremy Clarkson forced to ' kill off ' celebs on The Grand Tour
after BBC ' bans Top Gear rip - off star interviews ' '' . The Sun . Retrieved 26 November 2016 . Jump up ^ Lee , Kristen ( 16 November 2017 ) . `` ' Celebrity
Brain Crash ' Axed on the Grand Tour Season 2 '' . jalopnik.com . Retrieved 17 November 2017 . Jump up ^ McCreesh , Louise ( 27 November 2017 ) . `` The Grand
Tour season 2 confirms celebrity guests including The Hoff , Jack Bauer and Luke Evans '' . Digital Spy . Retrieved 28 November 2017 . Jump up ^ Hsieh , Tony (
20 November 2016 ) . `` The Grand Tour : 3 Mustangs from the Opening Sequence of the Premiere Episode '' . Grand Tour Nation . Retrieved 24 November 2016 . ^
Jump up to : `` Amazon Instant Video '' .
</item 5>

<item 6>
The Grand Tour (TV series)
The Sunday Times : Driving . Retrieved 14 December 2017 . The show has a new test track in Enstone , ... there planning problems when ( Clarkson ) tried once
before to use the Enstone track , back in Top Gear days ... The noisier testing of fast cars will continue to be done in Wiltshire at RAF Wroughton ...
nicknamed `` the Eboladrome '' Jump up ^ Reed , Jason ( 24 January 2018 ) . `` The Secret Story Behind The Grand Tour 's Celebrity Braincrash '' . Shifting
Lanes . Retrieved 26 January 2018 . ^ Jump up to : McCreesh , Louise ( 27 November 2017 ) . `` The Grand Tour season 2 confirms celebrity guests including
Kiefer Sutherland and David Hasselhoff '' . Retrieved 19 December 2017 . The celebrity guests this series include Luke Evans , Kiefer Sutherland , Hugh
Bonneville , Kevin Pietersen and Dominic Cooper ( ... ) Dynamo , Rory Mcllroy , Michael Ball and Alfie Boe will also appear in season two .
</item 6>

<item 7>
The Grand Tour (TV series)
1 Dry 14 David Hasselhoff 1 : 24.1 Kevin Pietersen 1 : 17.2 Dry 15 Brian Wilson 1 : 17.5 Hugh Bonneville 1 : 22.2 Dry 16 Casey Anderson 1 : 18.6 Michael Ball 1
: 23.3 Damp 17 Alfie Boe 1 : 24.4 Dominic Cooper 1 : 23.6 Wet 18 Bill Bailey 1 : 25.1 Luke Evans 1 : 21.3 Dry 19 Kiefer Sutherland 1 : 17.8 Bill Goldberg 1 :
20.4 Dry 20 Anthony Joshua 1 : 18.7 Nick Mason 1 : 21.3 Dry 21 Stewart Copeland 1 : 24.2 Dynamo 1 : 39.3 Snow 22 Penn & Teller 1 : 33.8 Rory McIlroy 1 : 21.9
Wet 23 Paris Hilton 1 : 25.8 Conversation Street ( edit ) Introduced by Hammond using a car analogy to often take a left and park on the street , the trio
discuss car news .
</item 7>

<item 8>
The Grand Tour (TV series)
19 6 `` Jaaaaaaaags '' None Luke Evans , Kiefer Sutherland 12 January 2018 ( 2018 - 01 - 12 ) The presenters head to the area around Grand Junction , Colorado ,
staying at the Gateway Canyons Resort & Spa , and eventually travelling to Telluride , Colorado with a variety of old Jaguar cars to establish the manufacturer
's reputation for reliability with Clarkson buying an XJR , May taking an XK8 convertible and Hammond bringing a Mark X . During the course of the trip the cars
are driven around a dirt track ( where Hammond 's car is badly damaged , and is replaced by an XJ6 ) , along the runway at Telluride Regional Airport , ( where
May 's car is destroyed , and is replaced by an XJS ) and down the slopes of the Telluride Ski Resort . At Hanging Flume overlook , Clarkson experiences a
traffic stop by a Mesa County police officer and reads the `` Her Britannic Majesty '' text from his British passport . Despite Clarkson 's car being the best ,
May admitting his car had poor brakes deemed the challenge pointless .
</item 8>

<item 9>
The Grand Tour (TV series)
1 Wet not shown Volkswagen Up ! GTI 1 : 39.7 Wet 18 Enstone ( edit ) Gravel course at Enstone Airfield introduced in Series 2 and used for Celebrity Face Off
For the second series , a new test track was created at Enstone Airfield , close to the fixed studio tent location , and which the production team had
previously tried to use for Top Gear . The main Eboladrome track was planned to continue to serve for the testing of loud and fast cars . Celebrity Face off (
edit ) Following complaints regarding the `` Celebrity Brain Crash '' segment in the first series , this segment was replaced by `` Celebrity Face Off '' . Two
celebrities join the show each week , with a similar background or connection and often from different countries , to see which one of the two is faster . The
car used for this segment is a Jaguar F - Type R - Dynamic coupe , racing around a half gravel , half asphalt track at Enstone . Prior to showing the lap
results , Clarkson interviews the guests . Lap board Celebrity Time Condition Episode Ricky Wilson 1 : 20.1 Dry 14 David Hasselhoff 1 : 24.
</item 9>

<item 10>
The Grand Tour (TV series)
ASIN B0731ZQ3R7. 1 . Past , present or future Missing or empty url = ( help ) ; access - date = requires url = ( help ) ^ Jump up to : Starkey , Adam ( 13
December 2017 ) . `` The Grand Tour season 2 : Richard Hammond is on crutches in episode 2 after horrifying crash '' . Metro . London . Retrieved 13 December
2017 . second episode titled The Falls Guys , Clarkson drives ... Ford GT from New York City to Niagara Falls , with James May making the same journey via
public transport with Hammond ... Celebrity Face Off ... cricketer Kevin Pietersen and baseball 's Brian Wilson Jump up ^ `` The Grand Tour Season 2 Episode 3
'' . 16 December 2017 . Retrieved 16 December 2017 . ^ Jump up to : `` Zadarska manekenka Silvija oduševila u ' The Grand Touru ' '' ( Zadar model Silvija
delights in The Grand Tour ) . 24sata ( in Croatian ) . 3 January 2018 .
</item 10>
=== END ITEM id=d2f6521da5f57b8c ===
